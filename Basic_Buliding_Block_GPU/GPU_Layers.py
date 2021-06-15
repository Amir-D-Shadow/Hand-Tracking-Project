import numpy as np
import math
from numba import cuda
import time

class ConvNet:
    
    @cuda.jit
    def conv_step_forward3D(W,img,b,Z,stride,xlim,ylim,zlim):

        """
        W -- (fH,fW,n_C_prev,n_C)
        img -- (n_H_prev,n_W_prev,n_C_prev)
        Z -- (n_H,n_W,n_C)
        """

        fH,fW,n_C_prev,n_C = W.shape
        
        n_H = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
        n_W = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
        n_C = cuda.threadIdx.z + cuda.blockIdx.z*cuda.blockDim.z

        if (n_H >= xlim) and (n_W >= ylim) and (n_C >= zlim):

            return

        W_filter = cuda.shared.array(W.shape)
        IMG = cuda.shared.array(img.shape)

        W_filter[:,:,:,:] = W[:,:,:,:]
        IMG[:,:,:] = img[:,:,:]

        #loop through filter
        for f in range(n_C):

            #loop through height
            for h in range(fH):

                #loop through width
                for w in range(fW):

                    #loop through channels
                    for c in range(n_C_prev):

                        IMG_H = n_H*stride+h
                        IMG_W = n_W*stride+w

                        Z[n_H][n_W][n_C] += W_filter[h,w,c,f]*IMG[IMG_H,IMG_W,c]+float(b[0,0,0,f])

                    
        cuda.syncthreads()


if __name__ == "__main__":

    obj = ConvNet()

    #GPU
    W = np.random.randn(5,5,3,16)
    b = np.random.randn(1,1,1,16)
    Img = np.random.randn(1080,1920,3)
    
    stride = 3
    n_H = int((1080-5)/stride)+1
    n_W = int((1920-5)/stride)+1
    n_C = 3
    
    Z = np.random.randn(n_H,n_W,16)
    
    threadsperblock = (8,8,2)

    blockspergrid_H = int(math.ceil(Z.shape[0]/threadsperblock[0]))
    blockspergrid_W = int(math.ceil(Z.shape[1]/threadsperblock[1]))
    blockspergrid_C = int(math.ceil(Z.shape[2]/threadsperblock[2]))

    W_device = cuda.to_device(W)
    Img_device = cuda.to_device(Img)
    Z_device = cuda.to_device(Z)
    
    gpu_time = time.time()
    obj.conv_step_forward3D(W,img)
    print(f"With GPU:{time.time()-gpu_time}")
    

    
