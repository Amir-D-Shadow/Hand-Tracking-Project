import numpy as np
import math
from numba import jit,cuda,float64,int64
import time

#Padding
def zero_padding(img,padH,padW):

        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad : amount of padding around each image on vertical and horizontal dimensions
        """

        return np.pad(img,((0,0),(padH,padH),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

def same_padding(img,stride,fH,fW):
    
    """
    img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    fH -- Kernel Size (height)
    fW -- Kernel Size (width)
    s -- Stride
    """
    n_H, n_W = img.shape[1],img.shape[2]
    padH = int(math.ceil(((n_H-1)*stride+fH-n_H)/2))
    padW = int(math.ceil(((n_W-1)*stride+fW-n_W)/2))

    img_same_pad = zero_padding(img,padH,padW)

    return img_same_pad,padH,padW

#Initialization
@jit(nopython=True)
def initialization_parameters(fH,fW,n_C_prev,n_C):

    W = (np.random.randn(fH,fW,n_C_prev,n_C).astype(np.float64))*np.sqrt(2/(n_C_prev+n_C))
    b = np.random.randn(1,1,1,n_C)

    return W,b

#Convolution Forward
def Conv_Forward3D_GPU(A_prev,W,b,stride,padH=0,padW=0,padding="Valid",threadsperblock=(8,8,2)):

    """
    A_prev -- (m, n_H_prev, n_W_prev, n_C_prev) 
    W -- (fH, fW, n_C_prev, n_C) 
    b -- (1, 1, 1, n_C)
    f -- kernel size
    """

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    fH, fW, n_C_prev, n_C = W.shape

    #Padding
    if padding == "Same":
        
        A_prev_pad,opadH,opadW = same_padding(A_prev,stride,fH,fW)

        n_H = n_H_prev
        n_W = n_W_prev

    elif padH != 0 or padW != 0:

        A_prev_pad = zero_padding(A_prev,padH,padW)

        n_H = int((n_H_prev+2*padH-fH)/stride)+1
        n_W = int((n_W_prev+2*padW-fW)/stride)+1
        opadH = padH
        opadW = padW

    else:

        A_prev_pad = A_prev.copy()

        n_H = int((n_H_prev-fH)/stride)+1
        n_W = int((n_W_prev-fW)/stride)+1
        opadH = 0
        opadW = 0

    #Set up
    shape_size = len(A_prev_pad.shape)
    Z = np.zeros((m,n_H,n_W,n_C))

    #Move memory to GPU
    A_prev_pad_device = cuda.to_device(A_prev_pad)
    W_device = cuda.to_device(W)
    b_device = cuda.to_device(b)
    Z_device = cuda.to_device(Z)

    #define blocks
    blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
    blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
    blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

    blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)
    cuda.synchronize()

    #convolution
    conv_step_forward3D_with_sample[blockspergrid,threadsperblock](W_device,A_prev_pad_device,b_device,Z_device,stride,n_H,n_W,n_C)
    cuda.synchronize()

    #GET RESULT
    Z = Z_device.copy_to_host()

    #Save cache for back propagation
    cacheL = (A_prev,W,b,stride,opadH,opadW)
    cuda.synchronize()

    return Z,cacheL


@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def conv_step_forward3D_with_sample(W,img,b,Z,stride,Hlim,Wlim,Clim):
    
    """
    W -- (fH,fW,n_C_prev,Channels)
    img -- (m,n_H_prev,n_W_prev,n_C_prev)
    b -- (1,1,1,Channels)
    Z -- (m,n_H,n_W,Channels)
    """
    fH,fW,n_C_prev,_ = W.shape
    m,n_H_prev,n_W_prev,n_C_prev = img.shape

    n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    n_W = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    n_C = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockIdx.z

    if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

        #Loop through each example
        for i in range(m):

            #Sum over volume

            #Loop through vertical axis
            for h in range(fH):

                #Loop through horizontal axis
                for w in range(fW):

                    #Loop through Channels (prev)
                    for c in range(n_C_prev):

                        IMG_H = n_H * stride + h
                        IMG_W = n_W * stride + w

                        Z[i,n_H,n_W,n_C] = Z[i,n_H,n_W,n_C] + W[h,w,c,n_C]*img[i,IMG_H,IMG_W,c]

            #Wait results
            cuda.syncthreads()

            #add bias
            Z[i,n_H,n_W,n_C] = Z[i,n_H,n_W,n_C] + float(b[0,0,0,n_C])

            #Wait results
            cuda.syncthreads()

                        
    
    
@cuda.jit("float64[:,:,:,:],float64[:,:,:],float64[:,:,:,:],float64[:,:,:],int64,int64,int64,int64")
def conv_step_forward3D(W,img,b,Z,stride,Hlim,Wlim,Clim):

    """
    W -- (fH,fW,n_C_prev,Channels)
    img -- (n_H_prev,n_W_prev,n_C_prev)
    b -- (1,1,1,Channels)
    Z -- (n_H,n_W,Channels)
    """

    fH,fW,n_C_prev,_ = W.shape
    n_H_prev,n_W_prev,n_C_prev = img.shape
    
    n_H = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    n_W = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
    n_C = cuda.threadIdx.z + cuda.blockIdx.z*cuda.blockDim.z

    if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

        #loop through height
        for h in range(fH):

            #loop through width
            for w in range(fW):

                #loop through channels
                for c in range(n_C_prev):

                    IMG_H = n_H*stride+h
                    IMG_W = n_W*stride+w

                    Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + W[h,w,c,n_C]*img[IMG_H,IMG_W,c]

        #wait until result come out
        cuda.syncthreads()

        #add bias
        Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + float(b[0,0,0,n_C])

        #wait until result come out
        cuda.syncthreads()


@cuda.jit("float64[:,:,:],float64[:,:],float64[:,:,:],float64[:,:,:],int64,int64,int64,int64")
def conv_step_forward2D(W,img,b,Z,stride,Hlim,Wlim,Clim):

    """
    W -- (fH,fW,Channels)
    img -- (n_H_prev,n_W_prev)
    b -- (1,1,Channels)
    Z -- (n_H,n_W,Channels)
    """

    fH,fW,_ = W.shape
    n_H_prev,n_W_prev = img.shape
    
    n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    n_W = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    n_C = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

        #loop through height
        for h in range(fH):

            #loop through width
            for w in range(fW):

                IMG_H = n_H * stride + h
                IMG_W = n_W * stride + w
                
                Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + W[h,w,n_C]*img[IMG_H,IMG_W]

        #wait result
        cuda.syncthreads()

        #Add Bias
        Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + float(b[0,0,n_C])

        #wait result
        cuda.syncthreads()

@cuda.jit("float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64,int64,int64")
def conv_step_forward1D(W,img,b,Z,stride,Hlim,Clim):

    """
    W -- (fH,Channels)
    img -- (n_H_prev,1)
    b -- (1,Channels)
    Z -- (n_H,Channels)
    """

    fH,_ = W.shape
    n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    n_C = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if (n_H < Hlim) and (n_C < Clim):

        for h in range(fH):

            IMG_H = n_H*stride + h

            Z[n_H,n_C] = Z[n_H,n_C] + W[h,n_C] * img[IMG_H]

        cuda.sycthreads()

        Z[n_H,n_C] = Z[n_H,n_C] + float(b[0,n_C])

        cuda.sycthread()

        


if __name__ == "__main__":

    #Test
    import Layers
    #4D
    #GPU
    W = np.random.randn(3,3,3,16).astype(np.float64)
    b = np.random.randn(1,1,1,16).astype(np.float64)
    img = np.random.randn(2,1080,1920,3)

    fH,fW,n_C_prev,n_C = W.shape
    m,n_H_prev,n_W_prev,n_C_prev = img.shape[1],img.shape[2]
    stride = 2

    n_H = int((n_H_prev-fH)/stride)+1
    n_W = int((n_W_prev-fW)/stride)+1

    Z = np.zeros(m,n_H,n_W,n_C)

    gpu_time = time.time()

    threadsperblock = (8,8,2)

    blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
    blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
    blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

    blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

    W_device = cuda.to_device(W)
    img_device = cuda.to_device(img)
    b_device = cuda.to_device(b)
    Z_device = cuda.to_device(Z)
    cuda.synchronize()

    conv_step_forward3D_with_sample[blockspergrid,threadsperblock](W_device,img_device,b_device,Z_device,stride,n_H,n_W,n_C)

    cuda.synchronize()
    k1 = Z_device.copy_to_host()
    cuda.synchronize()

    print(f"With GPU:{time.time()-gpu_time}")

    #CPU
    obj = Layers.ConvLayer()
    cpu_time = time.time()
    Z,_= obj.conv_forward(img,W,b,stride)            
    k2 = Z.copy()
    print(f"With CPU:{time.time()-cpu_time}")

    print(np.array_equal(k1.round(6),k2.round(6)))
    print(np.allclose(k1,k2))
     

    """
    #3D
    #GPU
    W = np.random.randn(3,3,3,16).astype(np.float64)
    b = np.random.randn(1,1,1,16).astype(np.float64)
    Img = np.random.randn(1,1080,1920,3).astype(np.float64)

    m,n_H_prev,n_W_prev,n_C_prev = Img.shape

    fH,fW = W.shape[0],W.shape[1]
    
    stride = 2
    n_H = int((n_H_prev-fH)/stride)+1
    n_W = int((n_W_prev-fW)/stride)+1
    n_C = 16
    
    Z = np.zeros((n_H,n_W,16))
    
    threadsperblock = (8,8,2)

    blockspergrid_H = int(math.ceil(Z.shape[0]/threadsperblock[0]))
    blockspergrid_W = int(math.ceil(Z.shape[1]/threadsperblock[1]))
    blockspergrid_C = int(math.ceil(Z.shape[2]/threadsperblock[2]))

    blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

    
    W_device = cuda.to_device(W)
    Img_device = cuda.to_device(Img[0,:,:,:])
    Z_device = cuda.to_device(Z)
    b_device = cuda.to_device(b)
    
    cuda.synchronize()
    
    gpu_time = time.time()
    conv_step_forward3D[blockspergrid,threadsperblock](W_device,Img_device,b_device,Z_device,stride,n_H,n_W,n_C)
    cuda.synchronize()
    k1 = Z_device.copy_to_host()
    print(f"With GPU:{time.time()-gpu_time}")
    

    #CPU
    obj = Layers.ConvLayer()
    cpu_time = time.time()
    Z,_= obj.conv_forward(Img,W,b,stride)            
    k2 = Z[0,:,:,:]
    print(f"With CPU:{time.time()-cpu_time}")

    print(np.array_equal(k1.round(6),k2.round(6)))
    """
    
    """
    #2D
    #GPU
    W = np.random.randn(3,3,16).astype(np.float64)
    b = np.random.randn(1,1,16).astype(np.float64)
    Img = np.random.randn(1080,1920).astype(np.float64)
    #W = np.ones((2,2,16))
    #b = np.ones((1,1,16))
    #Img = np.ones((108,192))

    n_H_prev,n_W_prev = Img.shape

    fH,fW = W.shape[0],W.shape[1]
    
    stride = 2
    n_H = int((n_H_prev-fH)/stride)+1
    n_W = int((n_W_prev-fW)/stride)+1
    n_C = 1
    
    Z = np.zeros((n_H,n_W,n_C))
    
    threadsperblock = (8,8,2)

    blockspergrid_H = int(math.ceil(Z.shape[0]/threadsperblock[0]))
    blockspergrid_W = int(math.ceil(Z.shape[1]/threadsperblock[1]))
    blockspergrid_C = int(math.ceil(Z.shape[2]/threadsperblock[2]))

    blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

    
    W_device = cuda.to_device(W)
    Img_device = cuda.to_device(Img)
    Z_device = cuda.to_device(Z)
    b_device = cuda.to_device(b)
    
    cuda.synchronize()
    
    gpu_time = time.time()
    conv_step_forward2D[blockspergrid,threadsperblock](W_device,Img_device,b_device,Z_device,stride,n_H,n_W,n_C)
    cuda.synchronize()
    k1 = Z_device.copy_to_host()
    print(f"With GPU:{time.time()-gpu_time}")

    #CPU
    obj = Layers.ConvLayer()
    cpu_time = time.time()
    #Get a sample
    #W = np.ones((2,2,1))
    #b = np.ones((1,1,1))
    a_prev_pad = Img.copy()
    Z = np.zeros((n_H,n_W,n_C))

    #Loop over vertical axis 
    for h in range(n_H):

         vert_start = h*stride
         vert_end = vert_start + fH
        
         #Loop over horizontal axis
         for w in range(n_W):

             hori_start = w*stride
             hori_end = hori_start + fW

             #Slice current sample
             a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end]

             #For each filter
             for c in range(n_C):

                 Wc = W[:,:,c]
                 bc = b[:,:,c]
                    
                 Z[h,w,c] =  np.sum(a_slice_prev*Wc)+float(bc)
                 
    k2 = Z.copy()
    print(f"With CPU:{time.time()-cpu_time}")

    print(np.array_equal(k1.round(6),k2.round(6)))
    """
    
