import cv2
import numpy as np
from numba import cuda,jit
import time
import math

@cuda.jit
def process_gpu(W,img):

    tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    tc = cuda.blockIdx.z*cuda.blockDim.z+cuda.threadIdx.z

    img[tx][ty][tc] = max(W[tx][ty][tc]*img[tx][ty][tc],0)

    
@jit
def process_cpu(W,img,dst):

    rows,cols,channels = img.shape

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):

                dst[i][j][c] = max(W[i][j][c]*dst[i][j][c],0)


def process_cpu_nothing(W,img,dst):

    rows,cols,channels = img.shape

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):

                dst[i][j][c] = max(W[i][j][c]*dst[i][j][c],0)



if __name__ == "__main__":

    img = cv2.imread("test.jpg")
    #img = cv2.resize(img,(256,512))
    #np.random.seed(1)
    #img = np.random.randn(256,256,3)
    rows,cols,channels=img.shape
    W = np.random.randn(rows,cols,channels)

    W_cpu = W.copy()
    W_gpu = W.copy()
    W_nothing = W.copy()

    dst_nothing = img.copy()
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    #cpu nothing
    cpu_start_time = time.time()
    cdImg  = process_cpu_nothing(W_nothing,img,dst_nothing)
    print(f"Nothing: {time.time()-cpu_start_time}")

    #cpu
    cpu_start_time = time.time()
    cdImg  = process_cpu(W_cpu,img,dst_cpu)
    print(f"Without GPU: {time.time()-cpu_start_time}")

    #gpu
    dImg = cuda.to_device(dst_gpu)
    gpu_W = cuda.to_device(W_gpu)
    
    threadsperblock = (8,8,3)
    
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols/threadsperblock[1]))
    blockspergrid_z = int(math.ceil(channels/threadsperblock[2]))
    
    blockspergrid = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    
    cuda.synchronize()
    
    gpu_start_time = time.time()
    process_gpu[blockspergrid,threadsperblock](gpu_W,dImg)
    cuda.synchronize()
    
    print(f"With GPU: {time.time()-gpu_start_time}")
    
    dst_gpu = dImg.copy_to_host()

    cuda.synchronize()

    print(np.array_equal(dst_cpu,dst_gpu))

    #save
    cv2.imwrite("result_cpu.jpg",dst_cpu)
    cv2.imwrite("result_gpu.jpg",dst_gpu)
