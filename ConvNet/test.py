import Layers
import numpy as np
from matplotlib.image import imread
import os
import time
import matplotlib.pyplot as plt
import test_utils

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
W = np.random.randn(3,3,3,128)
b = np.random.randn(1,1,1,128)
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

"""
#test conv backward
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
padH,padW,stride = 2,2,2
Z, cache_conv = obj.conv_forward(A_prev, W, b, stride,padH,padW)

dA, dW, db = obj.conv_backward(Z, cache_conv,lambda x:x)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))
"""

"""
#test pooling forward
#stride 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
stride,fH,fW = 1,3,3

obj = Layers.PoolingLayer()

A, cache = obj.Pooling(A_prev, stride,fH,fW)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()

A, cache = obj.Pooling(A_prev, stride,fH,fW, mode = "AVERAGE")

print("mode = average")
print("A.shape = " + str(A.shape))
print("A =\n", A)

#stride 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
stride,fH,fW = 2,3,3

A, cache = obj.Pooling(A_prev, stride,fH,fW)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()

A, cache = obj.Pooling(A_prev, stride,fH,fW, mode = "AVERAGE")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A =\n", A)

"""

"""
#test pooling backward
obj = Layers.PoolingLayer()

np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
fH,fW,stride = 2,2,1
A, cachePL = obj.Pooling_Forward(A_prev,stride,fH,fW)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = obj.Pooling_Backward(dA, cachePL, mode = "MAX")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = obj.Pooling_Backward(dA, cachePL, mode = "AVERAGE")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])

"""

"""
#Test Batch Layer (Train)
obj= Layers.Batch_Normalization_Layer()

Z = np.random.randn(4,5)
batch_para,running_para = obj.initialize_parameter(Z.shape)

Z_S,cacheBL = obj.batch_forward(Z,batch_para,running_para)

print("Z_S shape:{}".format(Z_S.shape))

gamma = batch_para["gamma"]
beta = batch_para["beta"]

bn_param = {}
bn_param['mode'] = "train"
bn_param['eps'] = 1e-10
bn_param['momentum']= 0.9

Z_C,cache= test_utils.batchnorm_forward(Z,gamma,beta,bn_param)

print("Z_C shape:{}".format(Z_C.shape))
"""

"""
#Test Batch Layer (Test)
obj= Layers.Batch_Normalization_Layer()
Z = np.random.randn(4,5)
batch_para,running_para = obj.initialize_parameter(Z.shape)

running_para["running_mean"] = 0.1
running_para["running_var"] = 0.1

Z_S,cacheBL = obj.batch_forward(Z,batch_para,running_para,mode="Test")

print("Z_S shape:{}".format(Z_S.shape))

gamma = batch_para["gamma"]
beta = batch_para["beta"]

bn_param = {}
bn_param['mode'] = "test"
bn_param['eps'] = 1e-10
bn_param['momentum']= 0.9
bn_param['running_mean'] = 0.1
bn_param['running_var'] = 0.1

Z_C,cache= test_utils.batchnorm_forward(Z,gamma,beta,bn_param)

print("Z_C shape:{}".format(Z_C.shape))
"""

"""
#Test Batch Layer backward 
obj= Layers.Batch_Normalization_Layer()

Z = np.random.randn(4,5)
dZ_S = np.random.randn(4,5)

batch_para,running_para = obj.initialize_parameter(Z.shape)

Z_S,cacheBL = obj.batch_forward(Z,batch_para,running_para)

dZ,dgamma,dbeta = obj.batch_backward(dZ_S,cacheBL)

print(f"dZ shape:{dZ.shape},dgamma shape:{dgamma.shape},dbeta shape:{dbeta.shape}")

gamma = batch_para["gamma"]
beta = batch_para["beta"]

bn_param = {}
bn_param['mode'] = "train"
bn_param['eps'] = 1e-10
bn_param['momentum']= 0.9

Z_C,cache= test_utils.batchnorm_forward(Z,gamma,beta,bn_param)

dZ_C, dgamma_C, dbeta_C = test_utils.batchnorm_backward(dZ_S,cache)

print(f"dZ_C shape:{dZ_C.shape},dgamma shape:{dgamma_C.shape},dbeta shape:{dbeta_C.shape}")
"""

"""
#Test dense forward
obj = Layers.Dense_layer()

A = np.random.randn(4,5)
parameters = obj.initialize_parameters(6,4)

Z,cacheDL = obj.dense_forward(A,parameters)

W = parameters["W"]
b = parameters["b"]


out, cache = test_utils.fc_forward(A,W,b)

print(f"Z shape:{Z.shape}")
print(f"out shape:{out.shape}")

"""

"""
#Test dense backward
obj = Layers.Dense_layer()

A = np.random.randn(4,5)
parameters = obj.initialize_parameters(6,4)

Z,cacheDL = obj.dense_forward(A,parameters)

dA = np.random.randn(6,5)

dW,db,dA_prev = obj.dense_backward(dA,cacheDL)

W = parameters["W"]
b = parameters["b"]

out, cache = test_utils.fc_forward(A,W,b)
dX, dWL, dbL = test_utils.fc_backward(dA,cache)

print(f"dW shape:{dW.shape},db shape:{db.shape},dA_prev shape:{dA_prev.shape}")
print(f"dWL shape:{dWL.shape},dbL shape:{dbL.shape},dXshape:{dX.shape}")
"""
