import numpy as np
import math
import time
from numba import jit

@jit(nopython=True)
def mult(arr1,arr2):

    return arr1*arr2

@jit(nopython=True)
def variance(arr):

    return np.var(arr)

@jit(nopython=True)
def mean(arr):

    return np.mean(arr)

@jit(nopython=True)
def dot(arr1,arr2):

    return np.dot(arr1,arr2)

if __name__ == "__main__":

    
    arr1 = np.random.randn(1080,1920,3)
    arr2 = np.random.randn(1080,1920,3)

    #no jit
    no_jit_time = time.time()
    k1 = arr1*arr2
    print(f"Without jit : {time.time()-no_jit_time}")

    #jit
    jit_time = time.time()
    k2 = mult(arr1,arr2)
    print(f"With jit : {time.time()-jit_time}")

    print(np.array_equal(k1,k2))
    

    """
    arr = np.random.randn(10,1080,1920,3)

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
    
