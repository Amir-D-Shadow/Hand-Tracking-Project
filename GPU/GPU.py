import cv2
import numpy as np
from numba import cuda,jit
import time
import math

@cuda.jit
def process_gpu(img,channels):

    tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

    for c in range(channels):

        color = img[tx,ty][c]*2.0+30

        if color >255:
            
            img[tx,ty][c] = 255

        elif color < 0:

            img[tx,ty][c] = 0

        else:

            img[tx,ty][c] = color


@jit
def process_cpu(img,dst):

    rows,cols,channels = img.shape

    for i in range(rows):
        for j in range(cols):
            for c in range(3):

                color = img[i,j][c]*2.0+30
                
                if color>255:
                    
                    dst[i,j][c] = 255

                elif color <0:

                    dst[i,j][c] = 0

                else:

                    dst[i,j][c] = color 


if __name__ == "__main__":

    #img = cv2.imread("test.jpg")
    img = np.random.randn(10000,10000,3)
    rows,cols,channels=img.shape
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    #cpu
    cpu_start_time = time.time()
    process_cpu(img,dst_cpu)
    print(f"Without GPU: {time.time()-cpu_start_time}")

    #gpu
    dImg = cuda.to_device(img)
    threadsperblock = (16,16)
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols/threadsperblock[1]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    cuda.synchronize()
    gpu_start_time = time.time()
    process_gpu[blockspergrid,threadsperblock](dImg,channels)
    cuda.synchronize()
    print(f"With GPU: {time.time()-gpu_start_time}")

    #print(np.array_equal(dst_cpu,dImg.copy_to_host()))
    dst_gpu = dImg.copy_to_host()

    #save
    cv2.imwrite("result_cpu.jpg",dst_cpu)
    cv2.imwrite("result_gpu.jpg",dst_gpu)
