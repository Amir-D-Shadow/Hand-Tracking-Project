import numpy as np


class ConvLayer:

    def __init__(self,units=512,padding="Same"):

        self.units = units
        self.padding = padding

    def initialize_parameter(self,fH,fW,n_C_prev,n_C):

        return np.random.randn(fH,fW,n_C_prev,n_C)*np.sqrt(2/(fH+fW))


    def zero_padding(self,img,padH,padW):

        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad : amount of padding around each image on vertical and horizontal dimensions
        """

        img_pad = np.pad(img,((0,0),(padH,padH),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

        return img_pad
    

    def same_padding(self,img,nH,nW,s,fH,fW):

        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        fH -- Kernel Size (height)
        fW -- Kernel Size (width)
        s -- Stride
        """

        padH = int(np.ceil(((nH-1)*s+fH-nH)/2))
        padW = int(np.ceil(((nW-1)*s+fW-nW)/2))

        img_pad = self.zero_padding(img,padH,padW)

        return img_pad,padH,padW


    def conv_step_forward(self,ASlice_prev,Wc,bc):

        """
        ASlice_prev -- slice of input data of shape (f, f, n_C_prev)
        Wc -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        bc -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        """

        #Important to apply float() to convert bL to scalar
        Z = np.sum(ASlice_prev*Wc)+float(bc)

        return Z


    def conv_forward(self,A_prev,W,b,stride,padH=0,padW=0,padding="Valid"):

        """
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (fH, fW, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        f -- kernel size
        """

        #Retrieve Shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (fH, fW, n_C_prev, n_C) = W.shape

        #Padding and determine the size of new layer 
        if padding == "Same":

            A_prev_pad,cpadH,cpadW= self.same_padding(A_prev,n_H_prev,n_W_prev,stride,fH,fW)

            n_H = n_H_prev
            n_W = n_W_prev
            
        elif padH != 0 or padW != 0:

            A_prev_pad = self.zero_padding(A_prev,padH,padW)
            
            n_H = int((n_H_prev+2*padH-fH)/stride)+1
            n_W = int((n_W_prev+2*padW-fW)/stride)+1

            cpadH = padH
            cpadW = padW

        else:

            A_prev_pad = A_prev.copy()

            n_H = int((n_H_prev-fH)/stride)+1
            n_W = int((n_W_prev-fW)/stride)+1

            cpadH = 0
            cpadW = 0

            
        #Set up Z
        Z = np.zeros((m,n_H,n_W,n_C))

        #Convolute Forward
        for i in range(m):

            #Get a sample
            a_prev_pad = A_prev_pad[i,:,:,:]

            #Loop over vertical axis 
            for h in range(n_H):

                 vert_start = h*stride
                 vert_end = vert_start + fH
                
                 #Loop over horizontal axis
                 for w in range(n_W):

                     hori_start = w*stride
                     hori_end = hori_start + fW

                     #Slice current sample
                     a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end,:]

                     #For each filter
                     for c in range(n_C):

                         Wc = W[:,:,:,c]
                         bc = b[:,:,:,c]
                            
                         Z[i,h,w,c] =  self.conv_step_forward(a_slice_prev,Wc,bc)


        cacheL = (A_prev,W,b,stride,cpadH,cpadW)

        return Z,cacheL


    def conv_backward(self,dA,cacheL,dev2dZ):

        """
        dA -- numpy array of shape (m,n_H,n_W,n_C)
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (fH, fW, n_C_prev, n_C)
        cacheL -- (A_prev,W,b,stride,cpadH,cpadW)
        dev2dZ -- function that calculate the derivative dZ
        """

        #Get informaton from cacheL
        A_prev,W,b,stride,cpadH,cpadW = cacheL

        #Get the shape of prev layer
        m,n_H_prev,n_W_prev,n_C_prev = A_prev.shape

        #Get the shape of current layer
        _,n_H,n_W,n_C = dA.shape

        #Get the shape of W
        (fH, fW, n_C_prev, n_C) = W.shape

        #Calculate dZ -- (m,n_H,n_W,n_C)
        dZ = dev2dZ(dA)

        #Initialize dA_prev,dW,db
        dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
        dW =  np.zeros((fH,fW,n_C_prev,n_C))
        db = np.zeros((1,1,1,n_C))

        #Pad A_prev to original shape which is used in convolution -- (m,n_H_prev+2p,n_W_prev+2p,n_C_prev)
        A_prev_pad = self.zero_padding(A_prev,cpadH,cpadW)

        #Pad dA_prev for back propagation -- (m,n_H_prev+2p,n_W_prev+2p,n_C_prev) : (shape of A_prev_pad is used in convolution --> dA_prev_pad will have same shape as A_prev_pad and dA_prev will treated at the end to get correct derivative of dA_prev)
        dA_prev_pad = self.zero_padding(dA_prev,cpadH,cpadW)

        #Back Propagation
        for i in range(m):

            #Get the sample
            a_prev_pad = A_prev_pad[i,:,:,:]

            #Loop through each filter
            for c in range(n_C):

                #Loop through vertical axis
                for h in range(n_H):

                    vert_start = h*stride
                    vert_end = vert_start + fH

                    #Loop through horizontal axis
                    for w in range(n_W):

                        hori_start = w*stride
                        hori_end = hori_start + fW

                        #Get the slice from a_prev_pad
                        a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end,:]

                        #dA_prev_pad
                        dA_prev_pad[i,vert_start:vert_end,hori_start:hori_end,:] += W[:,:,:,c]*dZ[i,h,w,c]

                        #dW
                        dW[:,:,:,c] += a_slice_prev*dZ[i,h,w,c]

                        #db
                        db[:,:,:,c] += dZ[i,h,w,c]
  
 

        if cpadH != 0 and cpadW != 0:

            dA_prev[:,:,:,:] = dA_prev_pad[:,cpadH:-cpadH,cpadW:-cpadW,:]

        elif cpadH != 0 and cpadW == 0:

            dA_prev[:,:,:,:] = dA_prev_pad[:,cpadH:-cpadH,:,:]

        elif cpadH == 0 and cpadW != 0:

            dA_prev[:,:,:,:] = dA_prev_pad[:,:,cpadW:-cpadW,:]

        else:

            dA_prev[:,:,:,:] = dA_prev_pad[:,:,:,:]

        return dA_prev,dW,db
        

        
        

class PoolingLayer:

    def __init__(self,stride=1,fH=1,fW=1):

        self.fH = fH
        self.fW = fW
        self.stride = stride

        
    def zero_padding(self,img,padH,padW):

        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad : amount of padding around each image on vertical and horizontal dimensions
        """

        img_pad = np.pad(img,((0,0),(padH,padH),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

        return img_pad


    def Pooling_Forward(self,A_prev,stride,fH,fW,mode="MAX",padding="Valid"):

        """
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        fH -- Kernel Size (height)
        fW -- Kernel Size (width)
        """

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        #Determine the size of new layer
        if padding == "Same":

            A_prev_pad,padH,padW = self.same_padding(A_prev,n_H_prev,n_W_prev,stride,fH,fW)

            n_H = n_H_prev
            n_W = n_W_prev

        else:

            A_prev_pad = A_prev.copy()
            
            n_H = int((n_H_prev-fH)/stride)+1
            n_W = int((n_W_prev-fW)/stride)+1

            padH = 0
            padW = 0

        #Set up Z
        Z = np.zeros((m,n_H,n_W,n_C_prev))

        #Convolute Forward
        for i in range(m):

            #Retrieve Sample
            a_prev_pad = A_prev_pad[i,:,:,:]

            #Loop over vertical axis
            for h in range(n_H):

                vert_start = h*stride
                vert_end = vert_start + fH

                #Loop over horizontal axis
                for w in range(n_W):

                    hori_start = w*stride
                    hori_end = hori_start + fW

                    #For each filter
                    for c in range(n_C_prev):

                        #Slice current sample
                        a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end,c]

                        if mode == "MAX":
                            
                            Z[i,h,w,c] = np.max(a_slice_prev)

                        elif mode == "AVERAGE":

                            Z[i,h,w,c] = np.mean(a_slice_prev)


        cachePL = (A_prev,stride,fH,fW,padH,padW)

        return Z,cachePL


    def Pooling_Backward(self,dA,cachePL,mode="MAX"):

        """
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A : (m,n_H,n_W,n_C)
        A_prev -- (m, n_H_prev, n_W_prev, n_C_prev)
        cachePL -- cache output from the forward pass of the pooling layer : (A_prev,stride,fH,fW,padH,padW)
        mode -- the pooling mode you would like to use, defined as a string 
    
        """
        A_prev,stride,fH,fW,padH,padW = cachePL
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        _,n_H,n_W,n_C = dA.shape

        #Initialize dA_prev
        dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))

        #Pad to get correct dimension
        if padH != 0 or padW != 0:
            
            A_prev_pad = self.zero_padding(A_prev,padH,padW)
            dA_prev_pad = self.zero_padding(dA_prev,padH,padW)

        else:

            A_prev_pad = A_prev.copy()
            dA_prev_pad = dA_prev.copy()

        #Loop through each example
        for i in range(m):

            a_prev_pad  = A_prev_pad[i,:,:,:]

            #Loop through each filter
            for c in range(n_C):

                #Loop through vertical direction
                for h in range(n_H):

                    vert_start = h*stride
                    vert_end = vert_start + fH

                    #Loop through horizontal direction
                    for w in range(n_W):

                        hori_start = w*stride
                        hori_end = hori_start + fW

                        #Get a slice from a particular filter
                        a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end,c]

                        if mode == "MAX":

                            dA_prev_pad[i,vert_start:vert_end,hori_start:hori_end,c] += self.MAX_POOLING_STEP_BACCKWARD(a_slice_prev)*dA[i,h,w,c]

                        elif mode == "AVERAGE":

                            dA_prev_pad[i,vert_start:vert_end,hori_start:hori_end,c] += self.AVG_POOLING_STEP_BACKWARD(dA[i,h,w,c],a_slice_prev)
                            

        if padH != 0 and padW != 0:

            dA_prev = dA_prev[:,padH:-padH,padW:-padW,:]

        elif padH != 0 and padW == 0:

            dA_prev = dA_prev_pad[:,padH:-padH,:,:]

        elif padH == 0 and padW != 0:

            dA_prev = dA_prev_pad[:,:,padW:-padW,:]

        else:

            dA_prev[:,:,:,:] = dA_prev_pad[:,:,:,:]


        return dA_prev

            

    def MAX_POOLING_STEP_BACCKWARD(self,A_slice_prev):

        """
        A_slice_prev -- (fH,fW)
        """

        #Bulid the mask
        mask = (A_slice_prev == np.max(A_slice_prev)).astype(int)

        return mask

    def AVG_POOLING_STEP_BACKWARD(self,dAij,A_slice_prev):

        """
        A_slice_prev -- (fH,fW)
        dAij -- scalar
        """

        #get shape
        fH,fW = A_slice_prev.shape

        #Calculate dA_slice_prev
        dA_slice_prev = np.ones((fH,fW))
        dA_slice_prev = dAij/(fH*fW)

        return dA_slice_prev

    

class Dense_layer:


    def __init__(self,activation,backward_activation,layer_dims=[10,5]):

        """
        layer_dims : list -- indicating how many hidden units in each layer (**notice that layer_dims[0] is dimensions of input layer not hidden layer)
        activation -- list -- a list containing functions (**notice that activation[0] is the activation function of first hidden layer)
        backward_activation -- a list containing function that calcualte derivative of activation of corresponding layer 
        """

        self.layer_dims = layer_dims[:]
        self.activation = activation[:]
        self.backward_activation = backward_activation[:]


    def initialize_parameters(self):

        length = len(self.layer_dims)

        parameters = {}

        for l in range(1,length):

            para_W = "W" + str(l)
            para_b = "b" + str(l)

            parameters[para_W] = np.random.randn(layers[l],layers[l-1])*np.sqrt(2/())
            parameters[para_b] = np.zeros((layers[l],1))

        return parameters


    def step_forward(self,a_prev,WL,bL,act_func_L):

        z = np.dot(WL,a_prev)+bL
        a_next = act_func_L(z)

        cache_L = (a_next,a_prev,z,WL,bL)

        return a_next,cache_L

    def forward_propagation(self,a0,parameters):

        length = len(self.layer_dims)

        a_prev = a0

        a = []
        cache = []

        for l in range(1,length):

            a_prev,cache_L = self.step_forward(a_prev,parameters["W"+str(l)],parameters["b"+str(l)],self.activation[l-1])
            
            a.append(a_prev)
            cache.append(cache_L)

        return a,cache


    def step_backward(self,da_next,backward_activation_L,cache_L):

        """
        cache_L: (a_next,a_prev,z,WL,bL)
        """
        a_next,a_prev,z,WL,bL = cache_L
        m = a_prev.shape[1]
        
        dZ = backward_activation_L(z) * da_next
        dW = np.dot(dZ,a_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        da_prev = np.dot(WL.T,dZ)

        return da_prev,dW,db


    def backward_propagation(self,dAL,cache):

        length = len(self.layer_dims)
        da_next = dAL

        gradients = {}

        for l in reversed(range(1,length)):

            grad_W = "dW"+str(l)
            grad_b = "db"+str(l)
            grad_A = "dA"+str(l-1)

            cache_L = cache[l-1]

            da_next,dW,db = self.step_backward(da_next,self.backward_activation[l-1],cache_L)

            gradients[grad_A] = da_next
            gradients[grad_W] = dW
            gradients[grad_b] = db

        return gradients

class Batch_Normalization_Layer:

    def initialize_parameter(self,shape,momentum=0.9):

        """
        first dimension represent number of examples
        shape : tuple
        """

        factor = 0
        beta_shape = []

        for i in shape[1:]:

            factor += i
            beta_shape.append(1)

        gamma = np.random.random_sample(shape[1:])*np.sqrt(2/(factor))

        beta = np.random.random_sample(tuple(beta_shape))

        running_para = {}
        batch_para = {}

        running_para["running_mean"] = 0
        running_para["running_var"] = 0
        running_para["momentum"] = momentum

        batch_para["gamma"] = gamma
        batch_para["beta"] = beta

        return batch_para,running_para

    def batch_forward(self,Z,batch_para,running_para,epsilon = 1e-10,mode="Train"):

        """
        first dimension represent number of examples
        default mode is "Train"
        """

        #Set up
        running_mean = running_para["running_mean"]
        running_var = running_para["running_var"]
        momentum = running_para["momentum"]

        gamma = batch_para["gamma"]
        beta = batch_para["beta"] 

        if mode == "Train":

            #calculate mean and variance
            mean_mu = np.mean(Z,axis=0,keepdims=True)
            var_sig = np.var(Z,axis=0,keepdims=True)

            #update running mean and variance
            running_para["running_mean"] = momentum*running_mean + (1-momentum)*mean_mu
            running_para["running_var"] = momentum*running_var + (1-momentum)*var_sig

            #calculate Z norm
            Z_NORM = (Z-mean_mu)/np.sqrt(var_sig+epsilon)

            #calculate output Z_S
            Z_S = gamma*Z_NORM + beta

            #save cache
            cacheBL = (Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_mu,var_sig)

        else:

            #calculate Z_norm
            Z_NORM = (Z-running_mean)/np.sqrt(running_var+epsilon)

            #calculate output Z_S
            Z_S = gamma*Z_NORM + beta

            cacheBL = ()
            
        
        return Z_S,cacheBL
    

    def batch_backward(self,dZ_S,cacheBL):

        """
        first dimension represent number of examples
        cacheBL contains (Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_mu,var_sig)
        """
        Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_mu,var_sig = cacheBL

        m = Z.shape[0]

        dZ_NORM = gamma*dZ_S
        dvar_sig = (-(Z-mean_mu)*dZ_NORM/(2*((var_sig+epsilon)**(1.5)))).sum(axis=0,keepdims=True)
        dmean_mu = (-dZ_NORM/np.sqrt(var_sig+epsilon)).sum(axis=0,keepdims=True)+dvar_sig*(-2/m)*(Z-mean_mu).sum(axis=0,keepdims=True)

        #print(f"dvar:{dvar_sig} , dmean:{dmean_mu}")
        dZ = dZ_NORM/np.sqrt(var_sig+epsilon)+dmean_mu/m+dvar_sig*2*(Z-mean_mu)/m
        dgamma = (Z_NORM*dZ_S).sum(axis=0,keepdims=True)
        dbeta = (dZ_S).sum(axis=0,keepdims=True)

        #print(f"dZ_NORM:{dZ_NORM}")

        return dZ,dgamma,dbeta

        
        
        

     

        

        
        
            

        

        





    
