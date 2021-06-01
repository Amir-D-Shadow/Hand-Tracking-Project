import numpy as np


class ConvLayer:

    def __init__(self,units=512,padding="Same"):

        self.units = units
        self.padding = padding


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


        cacheL = (A_prev,W,b,stride,(cpadH,cpadW))

        return Z,cacheL


class PoolingLayer:

    def __init__(self,stride=1,fH=1,fW=1):

        self.fH = fH
        self.fW = fW
        self.stride = stride


    def Pooling(self,A_prev,stride,fH,fW,mode="MAX",padding="Valid"):

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


        cachePL = (A_prev,stride,fH,fW)

        return Z,cachePL
                


        
        
            

        

        





    
