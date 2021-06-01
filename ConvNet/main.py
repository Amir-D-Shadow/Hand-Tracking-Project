import Layers
import numpy as np
from matplotlib.image import imread
import os
import time
import matplotlib.pyplot as plt

path = os.path.join(os.getcwd(),"Data","test.jpg")
img = imread(path)
obj = Layers.ConvLayer()

"""
a = np.random.randn(5,2,3,4)
b = np.random.randn(1,1,1)
c = np.sum(a)
d = float(b)
"""

"""
 #test zero padding
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = obj.zero_padding(x,2,2)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1,1])
print ("x_pad[1,1] =\n", x_pad[1,1])
"""

"""
#test conv forward and same padding
np.random.seed(1)
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(3,3,4,8)
b = np.random.randn(1,1,1,8)
padH,padW,stride = 1,1,2

Z, cache_conv = obj.conv_forward(A_prev, W, b,stride,padH,padW)

print("Z's mean =\n", np.mean(Z))
print("Z[3,2,1] =\n", Z[3,2,1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

np.random.seed(1)
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(2,5,4,8)
b = np.random.randn(1,1,1,8)
padH,padW,stride = 1,1,2

Z, cache_conv = obj.conv_forward(A_prev, W, b,stride,padH,padW)

print("Z's mean =\n", np.mean(Z))
print("Z[3,2,1] =\n", Z[3,2,1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

Z, cache_conv = obj.conv_forward(A_prev, W, b,stride,padding="Same")
print("Shape of Z:{}".format(Z.shape[1:3]))

A_prev = np.expand_dims(img/255,axis=0)
W = np.random.randn(1024,1880,3,1)
b = np.random.randn(1,1,1,2)
Z, cache_conv = obj.conv_forward(A_prev, W, b,stride,padding="Valid")
print("Shape of Z:{}".format(Z.shape[1:3]))
"""
"""
#test conv step forward
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = obj.conv_step_forward(a_slice_prev, W, b)
print("Z =", Z)
"""
