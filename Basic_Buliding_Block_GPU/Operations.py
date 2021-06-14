import numpy as np
import math
import time
from numba import jit,cuda

@jit(nopython=True)
def variance(arr):

    return np.var(arr)

@jit(nopython=True)
def mean(arr):

    return np.mean(arr)

@jit(nopython=True)
def dot(arr1,arr2):

    return np.dot(arr1,arr2)

@cuda.jit
def mult3D(arr1,arr2,xlim,ylim,zlim):

    x = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z*cuda.blockDim.z

    if (x < xlim) and (y < ylim) and (z < zlim):

        arr1[x][y][z] = arr1[x][y][z]*arr2[x][y][z]

@cuda.jit
def mult2D(arr1,arr2,xlim,ylim):

    x = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y

    if (x < xlim) and (y < ylim):

        arr1[x][y] = arr1[x][y]*arr2[x][y]

@cuda.jit
def mult1D(arr1,arr2,xlim):

    x = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

    if x < xlim:

        arr1[x] = arr1[x]*arr2[x]
        
  
def elementwise_mult(arr1,arr2,threads=8,last_thread = 2):

    N = len(arr1.shape)
    
    arr1_device = cuda.to_device(arr1)
    arr2_device = cuda.to_device(arr2)

    if N == 3:

        #Get sahpe
        x_dim,y_dim,z_dim = arr1.shape
        
        #Decide GPU Arrangement
        threadsperblock = (threads,threads,last_thread)

        x_blockspergrid = int(math.ceil(x_dim/threads))
        y_blockspergrid = int(math.ceil(y_dim/threads))
        z_blockspergrid = int(math.ceil(z_dim/last_thread))

        blockspergrid = (x_blockspergrid,y_blockspergrid,z_blockspergrid)

        #Wait CPU
        cuda.synchronize()

        #Start multiplication
        mult3D[blockspergrid,threadsperblock](arr1_device,arr2_device,x_dim,y_dim,z_dim)

        #Wait GPU
        cuda.synchronize()
        
        return arr1_device.copy_to_host()

    elif N ==2:

        #Get sahpe
        x_dim,y_dim = arr1.shape
        
        #Decide GPU Arrangement
        threadsperblock = (threads,last_thread)

        x_blockspergrid = int(math.ceil(x_dim/threads))
        y_blockspergrid = int(math.ceil(y_dim/last_thread))

        blockspergrid = (x_blockspergrid,y_blockspergrid)

        #Wait CPU
        cuda.synchronize()

        #Start multiplication
        mult2D[blockspergrid,threadsperblock](arr1_device,arr2_device,x_dim,y_dim)

        #Wait GPU
        cuda.synchronize()
        
        return arr1_device.copy_to_host()

    elif N == 1:

        #Get sahpe
        x_dim = arr1.shape[0]
        
        #Decide GPU Arrangement
        threadsperblock = (threads)

        x_blockspergrid = int(math.ceil(x_dim/threads))

        blockspergrid = (x_blockspergrid)

        #Wait CPU
        cuda.synchronize()

        #Start multiplication
        mult1D[blockspergrid,threadsperblock](arr1_device,arr2_device,x_dim)

        #Wait GPU
        cuda.synchronize()
        
        return arr1_device.copy_to_host()


if __name__ == "__main__":

    """
    arr = np.random.randn(100,1080,1920,3)

    #mean
    #no jit
    no_jit_time = time.time()
    k1 = np.mean(arr)
    print(f"Without jit: {time.time()-no_jit_time}")

    #jit
    jit_time = time.time()
    k2 = mean(arr)
    print(f"With jit: {time.time()-jit_time}")

    #variance
    #no jit
    no_jit_time = time.time()
    k1 = np.var(arr)
    print(f"Without jit: {time.time()-no_jit_time}")

    #jit
    jit_time = time.time()
    k2 = variance(arr)
    print(f"With jit: {time.time()-jit_time}")
    """
    """
    arr1 = np.random.randn(1080,1920)
    arr2 = np.random.randn(1080,1920)

    #dot
    #no jit
    no_jit_time = time.time()
    k1 = np.dot(arr1.T,arr2)
    print(f"Without jit: {time.time()-no_jit_time}")

    #jit
    jit_time = time.time()
    k2 = np.dot(arr1.T,arr2)
    print(f"With jit: {time.time()-jit_time}")

    """
    """
    #mult
    arr1 = np.random.randn(1080,1920,3)
    arr2 = np.random.randn(1080,1920,3)

    #4D
    #cpu
    k1 = np.zeros_like(arr1)
    cpu_time = time.time()
    for i in range(arr1.shape[0]):
        
        k1[i,:,:,:]  = arr1[i,:,:,:]*arr2[i,:,:,:]
        
    print(f"cpu: {time.time()-cpu_time}")

    #gpu
    k2 = np.zeros_like(arr1)
    gpu_time = time.time()
    for i in range(arr1.shape[0]):
        
        k2[i,:,:,:] = elementwise_mult(arr1[i,:,:,:],arr2[i,:,:,:])
        
    print(f"gpu: {time.time()-gpu_time}")

    print(np.array_equal(k1,k2))
    """
    """
    #3D
    #cpu
    cpu_time = time.time()
    k1 = arr1*arr2
    print(f"cpu: {time.time()-cpu_time}")

    #gpu
    gpu_time = time.time()
    k2 = elementwise_mult(arr1,arr2)
    print(f"gpu: {time.time()-gpu_time}")

    print(np.array_equal(k1,k2))
    """

    
