import numpy as np
import math
from numba import cuda,float64,int64
import time
import Layers

@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def conv_step_forward4D(W,img,b,Z,stride,xlim,ylim,zlim):
    
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

    if (n_H < xlim) and (n_W < ylim) and (n_C < zlim):

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
def conv_step_forward3D(W,img,b,Z,stride,xlim,ylim,zlim):

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

    if (n_H < xlim) and (n_W < ylim) and (n_C < zlim):

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
def conv_step_forward2D(W,img,b,Z,stride,xlim,ylim,zlim):

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

    if (n_H < xlim) and (n_W < ylim) and (n_C < zlim):

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
def conv_step_forward1D(W,img,b,Z,stride,xlim,ylim):

    """
    W -- (fH,Channels)
    img -- (n_H_prev,1)
    b -- (1,Channels)
    Z -- (n_H,Channels)
    """

    fH,_ = W.shape
    n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    n_C = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if (n_H < xlim) and (n_C < ylim):

        for h in range(fH):

            IMG_H = n_H*stride + h

            Z[n_H,n_C] = Z[n_H,n_C] + W[h,n_C] * img[IMG_H]

        cuda.sycthreads()

        Z[n_H,n_C] = Z[n_H,n_C] + float(b[0,n_C])

        cuda.sycthread()

        


if __name__ == "__main__":

    #Test

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

    conv_step_forward4D[blockspergrid,threadsperblock](W_device,img_device,b_device,Z_device,stride,n_H,n_W,n_C)

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
    
    """
    #W_device = cuda.device_array_like(W)
    #Img_device = cuda.device_array_like(Img[0,:,:,:])
    #Z_device = cuda.device_array_like(Z)
    #b_device = cuda.device_array_like(b)
    """
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
    
