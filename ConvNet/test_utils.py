import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        std = np.sqrt(sample_var + eps)
        x_centered = x - sample_mean
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        
        cache = (x_norm, x_centered, std, gamma)
        
    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = ()
    
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    Use a computation graph for batch normalization and propagate gradients 
    backward through intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    N = dout.shape[0]
    x_norm, x_centered, std, gamma = cache
    dgamma = (dout * x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)
    
    dx_norm = dout * gamma
    dx_centered = dx_norm / std
    dmean = -(dx_centered.sum(axis=0) + 2/N * x_centered.sum(axis=0))
    dstd = (dx_norm * x_centered * -std**(-2)).sum(axis=0)
    dvar = dstd / 2 / std
    dx = dx_centered + (dmean + dvar * 2 * x_centered) / N

    #print(f"dvar:{dvar} , dmean:{dmean}")
    #print(f"dx_norm:{dx_norm}")

    return dx, dgamma, dbeta

def fc_forward(X, W, b):
    out = np.dot(W,X) + b
    cache = (W, X)
    return out, cache

def fc_backward(dout, cache):
    W, h = cache

    dW = np.dot(dout,h.T)
    db = np.sum(dout, axis=1)
    dX = np.dot(W.T,dout)

    return dX, dW, db
