import numpy as np

def sigmoid(z):

    z = np.where(z>=0,np.minimum(z,1e3),np.maximum(z,-1e3))

    return np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))


def dev_sigmoid(z):

    y = sigmoid(z)

    return y*(1-y)

def relu(z):

    return np.maximum(z,0)

def dev_relu(z):

    dA = np.zeros(z.shape)
    dA[z>=0] = 1

    return dA

def tanh(z):

    return np.tanh(z)

def dev_tanh(z):

    y = tanh(z)

    return (1 - y**2)
    
