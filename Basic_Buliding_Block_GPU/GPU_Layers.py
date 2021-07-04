import numpy as np
import math
from numba import jit,cuda,float64,int64
import time

#Padding
def zero_padding(img,padH,padW):

        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad : amount of padding around each image on vertical and horizontal dimensions
        """

        return np.pad(img,((0,0),(padH,padH),(padW,padW),(0,0)),mode="constant",constant_values=(0,0))

def same_padding(img,stride,fH,fW):
    
        """
        img : numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        fH -- Kernel Size (height)
        fW -- Kernel Size (width)
        s -- Stride
        """
        n_H, n_W = img.shape[1],img.shape[2]
        padH = int(math.ceil(((n_H-1)*stride+fH-n_H)/2))
        padW = int(math.ceil(((n_W-1)*stride+fW-n_W)/2))

        img_same_pad = zero_padding(img,padH,padW)

        return img_same_pad,padH,padW

#Initialization
@jit(nopython=True)
def initialization_parameters(fH,fW,n_C_prev,n_C):

        W = (np.random.randn(fH,fW,n_C_prev,n_C).astype(np.float64))*np.sqrt(2/(n_C_prev+n_C))
        b = (np.random.randn(1,1,1,n_C))

        return W,b

#ConvLayer

#Convolution Forward
def Conv_Forward3D_GPU(A_prev,W,b,stride,padH=0,padW=0,padding="Valid",threadsperblock=(8,8,8)):

        """
        A_prev -- (m, n_H_prev, n_W_prev, n_C_prev) 
        W -- (fH, fW, n_C_prev, n_C) 
        b -- (1, 1, 1, n_C)
        """

        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        fH, fW, n_C_prev, n_C = W.shape

        #Padding
        if padding == "Same":

                A_prev_pad,opadH,opadW = same_padding(A_prev,stride,fH,fW)

                n_H = n_H_prev
                n_W = n_W_prev

        elif padH != 0 or padW != 0:

                A_prev_pad = zero_padding(A_prev,padH,padW)

                n_H = int((n_H_prev+2*padH-fH)/stride)+1
                n_W = int((n_W_prev+2*padW-fW)/stride)+1
                opadH = padH
                opadW = padW

        else:

                A_prev_pad = A_prev.copy()

                n_H = int((n_H_prev-fH)/stride)+1
                n_W = int((n_W_prev-fW)/stride)+1
                opadH = 0
                opadW = 0

        #Set up
        Z = np.zeros((m,n_H,n_W,n_C))

        #Move memory to GPU
        W_device = cuda.to_device(W)
        b_device = cuda.to_device(b)

        #Create stream
        stream_list = []
        for i in range(m):

            stream_list.append(cuda.stream())

        #define blocks for H and W
        blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)
        #cuda.synchronize()

        #convolution
        for s in range(m):

            #Move memory to GPU    
            Z_i = (Z[s,:,:,:]).copy()
            Z_device = cuda.to_device(Z_i,stream = stream_list[s])
            
            A_prev_pad_i = (A_prev_pad[s,:,:,:]).copy()
            A_prev_pad_device = cuda.to_device(A_prev_pad_i,stream = stream_list[s])
            
            #calculation
            conv_step_forward3D[blockspergrid,threadsperblock,stream_list[s]](W_device,A_prev_pad_device,b_device,Z_device,stride,n_H,n_W,n_C)
            cuda.synchronize()

            #GET RESULT
            Z[s,:,:,:] = Z_device.copy_to_host(stream = stream_list[i])
            

        #Save cache for back propagation
        cacheL = (A_prev,W,b,stride,opadH,opadW)
        cuda.synchronize()

        return Z,cacheL


@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def conv_step_forward3D_with_sample(W,img,b,Z,stride,Hlim,Wlim,Clim):
    
        """
        W -- (fH,fW,n_C_prev,Channels)
        img -- (m,n_H_prev,n_W_prev,n_C_prev)
        b -- (1,1,1,Channels)
        Z -- (m,n_H,n_W,Channels)
        """
        fH,fW,n_C_prev,_ = W.shape
        m,n_H_prev,n_W_prev,n_C_prev = img.shape

        n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        n_W = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        n_C = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockIdx.z

        if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

                #Loop through each example
                for i in range(m):

                    #Sum over volume

                    #Loop through vertical axis
                    for h in range(fH):

                        #Loop through horizontal axis
                        for w in range(fW):

                            #Loop through Channels (prev)
                            for c in range(n_C_prev):

                                IMG_H = n_H * stride + h
                                IMG_W = n_W * stride + w

                                Z[i,n_H,n_W,n_C] = Z[i,n_H,n_W,n_C] + W[h,w,c,n_C]*img[i,IMG_H,IMG_W,c]

                    #Wait results
                    cuda.syncthreads()

                    #add bias
                    Z[i,n_H,n_W,n_C] = Z[i,n_H,n_W,n_C] + float(b[0,0,0,n_C])

                    #Wait results
                    cuda.syncthreads()

                        
    
    
@cuda.jit("float64[:,:,:,:],float64[:,:,:],float64[:,:,:,:],float64[:,:,:],int64,int64,int64,int64")
def conv_step_forward3D(W,img,b,Z,stride,Hlim,Wlim,Clim):

        """
        W -- (fH,fW,n_C_prev,Channels)
        img -- (n_H_prev,n_W_prev,n_C_prev)
        b -- (1,1,1,Channels)
        Z -- (n_H,n_W,Channels)
        """

        fH,fW,n_C_prev,_ = W.shape
        n_H_prev,n_W_prev,n_C_prev = img.shape

        n_H = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
        n_W = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
        n_C = cuda.threadIdx.z + cuda.blockIdx.z*cuda.blockDim.z

        if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

                #loop through height
                for h in range(fH):

                    #loop through width
                    for w in range(fW):

                        #loop through channels
                        for c in range(n_C_prev):

                            IMG_H = n_H*stride+h
                            IMG_W = n_W*stride+w

                            Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + W[h,w,c,n_C]*img[IMG_H,IMG_W,c]

                #wait until result come out
                cuda.syncthreads()

                #add bias
                Z[n_H,n_W,n_C] = Z[n_H,n_W,n_C] + float(b[0,0,0,n_C])

                #wait until result come out
                cuda.syncthreads()


#Convolution Backward
def Conv_backward3D_GPU(dZ,cacheL,threadsperblock=(8,8,8)):

        """
        dA -- (m,n_H,n_W,n_C)
        cacheL -- (A_prev,W,b,stride,opadH,opadW)
        """
        #Get informaton from cacheL
        A_prev,W,b,stride,opadH,opadW = cacheL

        #Get the shape of prev layer
        m,n_H_prev,n_W_prev,n_C_prev = A_prev.shape
        
        #Get the shape of current layer
        m,n_H,n_W,n_C = dZ.shape

        #Get the shape of filter
        fH, fW, n_C_prev,n_C = W.shape

        #Get the shape of b
        bH,bW,b_C_prev,n_C = b.shape
        
        #Set up
        dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
        dW = np.zeros((fH,fW,n_C_prev,n_C))
        db = np.zeros((1,1,1,n_C))

        #move memory to GPU A_prev_pad,dA_prev_pad
        if opadH == 0 and opadW == 0:

                #A_prev -- (m,n_H_prev,n_W_prev,n_C_prev)
                A_prev_pad = A_prev.copy()
                #dA_prev -- (m,n_H_prev,n_W_prev,n_C_prev)
                dA_prev_pad = dA_prev.copy()
        else:
                
                A_prev_pad = zero_padding(A_prev,opadH,opadW)
                dA_prev_pad = zero_padding(dA_prev,opadH,opadW)

        A_prev_pad_device = cuda.to_device(A_prev_pad)
        dA_prev_pad_device = cuda.to_device(dA_prev_pad)

        #define stream 
        segment_size = threadsperblock[-1]
        number_of_streams = int(math.ceil(n_C/segment_size))

        stream_list = []
        for i in range(number_of_streams):

                stream_list.append(cuda.stream())
                
        
        #blockspergrid for dA_prev_pad
        m,n_H_prev_pad,n_W_prev_pad,n_C_prev = dA_prev_pad.shape
        
        blockspergrid_H = int(math.ceil(n_H_prev_pad/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(n_W_prev_pad/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(n_C_prev/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)
       
        #backward dA_prev_pad
        for s in range(number_of_streams):

                #move memory to GPU
                W_device = cuda.to_device((W[:,:,:,(s*segment_size):((s+1)*segment_size)]).copy(),stream = stream_list[s])                
                dZ_device = cuda.to_device((dZ[:,:,:,(s*segment_size):((s+1)*segment_size)]).copy(),stream = stream_list[s])

                #calculation
                conv_step_backward3D_dA_prev_pad[blockspergrid,threadsperblock,stream_list[s]](dA_prev_pad_device,W_device,dZ_device,stride,n_H_prev_pad,n_W_prev_pad,n_C_prev)
                
        
        #Get Result dA_prev
        cuda.synchronize()
        dA_prev_pad = dA_prev_pad_device.copy_to_host()
        
        """
        #calculate dA_prev_pad
        conv_step_backward3D_dA_prev_pad_jit(dA_prev_pad,W,dZ,stride)
        """
        
        if opadH != 0 and opadW != 0:

                dA_prev[:,:,:,:] = dA_prev_pad[:,opadH:-opadH,opadW:-opadW,:]

        elif opadH != 0 and opadW == 0:

                dA_prev[:,:,:,:] = dA_prev_pad[:,opadH:-opadH,:,:]

        elif opadH ==0 and opadW != 0:

                dA_prev[:,:,:,:] = dA_prev_pad[:,:,opadW:-opadW,:]

        else:
                dA_prev = dA_prev_pad

        #blockspergrid for dW
        blockspergrid_H = int(math.ceil(fH/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(fW/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(n_C_prev/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

        #backward dW
        for s in range(number_of_streams):

          dW_device = cuda.to_device(dW[:,:,:,(s*segment_size):((s+1)*segment_size)].copy(),stream=stream_list[s])
          dZ_device = cuda.to_device(dZ[:,:,:,(s*segment_size):((s+1)*segment_size)].copy(),stream=stream_list[s])

          conv_step_backward3D_dW[blockspergrid,threadsperblock,stream_list[s]](A_prev_pad_device,dW_device,dZ_device,stride,fH,fW,n_C_prev)
          cuda.synchronize()

          dW[:,:,:,(s*segment_size):((s+1)*segment_size)] = dW_device.copy_to_host(stream=stream_list[s])

        #blockspergrid for db
        blockspergrid_H = int(math.ceil(bH/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(bW/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(b_C_prev/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

        cuda.synchronize()
        
        #backward db
        for s in range(number_of_streams):

          db_device = cuda.to_device(db[:,:,:,(s*segment_size):((s+1)*segment_size)].copy(),stream=stream_list[s])
          dZ_device = cuda.to_device(dZ[:,:,:,(s*segment_size):((s+1)*segment_size)].copy(),stream=stream_list[s])

          conv_step_backward3D_db[blockspergrid,threadsperblock,stream_list[s]](db_device,dZ_device,bH,bW,b_C_prev)
          cuda.synchronize()

          db[:,:,:,(s*segment_size):((s+1)*segment_size)] = db_device.copy_to_host(stream=stream_list[s])

        cuda.synchronize()

        return dA_prev,dW,db
    

@jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64",nopython=True)
def conv_step_backward3D_dA_prev_pad_jit(dA_prev_pad,W,dZ,stride):

    fH,fW,n_C_prev,n_C = W.shape
    m,n_H,n_W,n_C = dZ.shape

    for i in range(m):

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

                    #dA_prev_pad
                    dA_prev_pad[i,vert_start:vert_end,hori_start:hori_end,:] += W[:,:,:,c]*dZ[i,h,w,c]
                        


@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def conv_step_backward3D_dA_prev_pad(dA_prev_pad,W,dZ,stride,Hlim,Wlim,Clim):

  """
  dA_prev_pad -- (m,n_H_prev,n_W_prev,n_C_prev)
  W -- (fH,fW,n_C_prev,n_C)
  dZ -- (m,n_H,n_W,n_C)
  """

  IMG_H = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  IMG_W = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  IMG_C_prev = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (IMG_H < Hlim) and (IMG_W < Wlim) and (IMG_C_prev < Clim):

    fH,fW,n_C_prev,number_of_filters = W.shape
    m,n_H,n_W,number_of_filters = dZ.shape

    #method 1
    #find the corresponding components
    #loop through different example
    for i in range(m):

      #loop through different filters
      for nc in range(number_of_filters):

        for h in range(fH):

          for w in range(fW):

              nh = (IMG_H - h) / stride
              nw = (IMG_W - w) / stride

              #check if nw,nh are non-negative integers
              if (nh >= 0) and (nw >= 0) and ((nh-int(nh)) == 0 ) and ((nw-int(nw)) == 0 ):

                  #convert back to int as nh and nw become float after division
                  nh = int(nh)
                  nw = int(nw)

                  dA_prev_pad[i,IMG_H,IMG_W,IMG_C_prev] = dA_prev_pad[i,IMG_H,IMG_W,IMG_C_prev] + W[h,w,IMG_C_prev,nc] * dZ[i,nh,nw,nc]
                  
    """
    #method 2
    #find the corresponding components
    #loop through different example
    for i in range(m):

      #loop through different filters
      for nc in range(number_of_filters):
        for nh in range(n_H):

          for nw in range(n_W):

            for h in range(fH):

              for w in range(fW):

                cor_H = nh * stride + h
                cor_W = nw * stride + w


                if (cor_H == IMG_H) and (cor_W == IMG_W):

                  dA_prev_pad[i,IMG_H,IMG_W,IMG_C_prev] = dA_prev_pad[i,IMG_H,IMG_W,IMG_C_prev] + W[h,w,IMG_C_prev,nc] * dZ[i,nh,nw,nc]

    """
                  
@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def conv_step_backward3D_dW(A_prev_pad,dW,dZ,stride,Hlim,Wlim,Clim):

  """
  A_prev_pad -- (m,n_H_prev,n_W_prev,n_C_prev)
  dW -- (fH,fW,n_C_prev,segment_size)
  dZ -- (m,n_H,n_W,segemnt_size)
  """

  h = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  w = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  n_C_prev = cuda.threadIdx.z +cuda.blockDim.z*cuda.blockIdx.z

  if (h < Hlim) and (w < Wlim) and (n_C_prev < Clim):

    m,nH,nW,segemnt_size = dZ.shape
    #method 1
    
    for i in range(m):

      for n_h in range(nH):

        for n_w in range(nW):

            for n_c in range(segemnt_size):

              IMG_H = n_h*stride + h
              IMG_W = n_w*stride + w

              dW[h,w,n_C_prev,n_c] = dW[h,w,n_C_prev,n_c] + A_prev_pad[i,IMG_H,IMG_W,n_C_prev]*dZ[i,n_h,n_w,n_c]
    
    """
    #method 2
    for i in range(m):

      for n_c in range(segemnt_size):

        for n_h in range(nH):
          
          #get n_w tail
          n_w_tail = nW - 1

          for n_w in range(nW):
            
            if n_w > n_w_tail:

              break

            if (n_w == n_w_tail):
              
              IMG_H = n_h*stride + h
              IMG_W = n_w*stride + w

              dW[h,w,n_C_prev,n_c] = dW[h,w,n_C_prev,n_c] + A_prev_pad[i,IMG_H,IMG_W,n_C_prev]*dZ[i,n_h,n_w,n_c]

            else:

              #calculate head 
              IMG_H = n_h*stride + h
              IMG_W = n_w*stride + w

              dW[h,w,n_C_prev,n_c] = dW[h,w,n_C_prev,n_c] + A_prev_pad[i,IMG_H,IMG_W,n_C_prev]*dZ[i,n_h,n_w,n_c]

              #calculate tail
              #IMG_H = n_h*stride + h
              IMG_W = n_w_tail*stride + w

              dW[h,w,n_C_prev,n_c] = dW[h,w,n_C_prev,n_c] + A_prev_pad[i,IMG_H,IMG_W,n_C_prev]*dZ[i,n_h,n_w_tail,n_c]

            n_w_tail = n_w_tail - 1
              
          
    """
    
@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64")
def conv_step_backward3D_db(db,dZ,Hlim,Wlim,Clim):

  """
  db -- (1,1,1,n_C)
  dZ -- (m,n_H,n_W,n_C)
  """

  x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  z = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (x < Hlim) and (y < Wlim) and (z < Clim):

    m,nH,nW,segemnt_size = dZ.shape

    for i in range(m):

      for n_h in range(nH):

        for n_w in range(nW):

          for n_c in range(segemnt_size):

            db[x,y,z,n_c] = db[x,y,z,n_c] + dZ[i,n_h,n_w,n_c]


#MaxPooling

#Forward
def MaxPooling_Forward3D(Z_prev,kernel_size,stride,threadsperblock=(8,8,8),padH=0,padW=0,padding = "Valid"):

   """
   Z_prev -- (m,n_H_prev,n_W_prev,n_C_prev)
   Kernel_size -- tuple(fH,fW)
   """

   m,n_H_prev,n_W_prev,n_C_prev = Z_prev.shape
   fH,fW = kernel_size

   #Padding
   if padH == 0 and padW == 0:

      Z_prev_pad = Z_prev.copy()

      n_H = int((n_H_prev-fH)/stride) + 1
      n_W = int((n_W_prev-fW)/stride) + 1
      n_C = n_C_prev

      opadH = 0
      opadW = 0

   elif padding == "Same":

        Z_prev_pad,opadH,opadW = same_padding(Z_prev,stride,fH,fW)

        n_H = n_H_prev
        n_W = n_W_prev
        n_C = n_C_prev
        
   elif padH != 0 or padW != 0:

      Z_prev_pad = zero_padding(Z_prev,padH,padW)

      n_H = int((n_H_prev+2*padH-fH)/stride) + 1
      n_W = int((n_W_prev+2*padW-fW)/stride) + 1
      n_C = n_C_prev

      opadH = padH
      opadW = padW

   #Set up
   Z = np.zeros((m,n_H,n_W,n_C))

   #Define Stream
   stream_list = []
   number_of_streams = m

   for i in range(number_of_streams):

      stream_list.append(cuda.stream())

   #Define Blocks per grid
   blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
   blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
   blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

   blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

   #MaxPooling Forward
   for s in range(number_of_streams):

      Z_prev_pad_device = cuda.to_device(Z_prev_pad[s,:,:,:].copy(),stream=stream_list[s])
      Z_device = cuda.to_device(Z[s,:,:,:].copy(),stream=stream_list[s])

      MaxPooling_Step_Forward3D[blockspergrid,threadsperblock,stream_list[s]](Z_prev_pad_device,Z_device,fH,fW,stride,n_H,n_W,n_C)
      cuda.synchronize()

      Z[s,:,:,:] = Z_device.copy_to_host(stream=stream_list[s])

   cacheL = (Z_prev,Z,kernel_size,stride,opadH,opadW)

   return Z,cacheL
      

@cuda.jit("float64[:,:,:],float64[:,:,:],int64,int64,int64,int64,int64,int64")
def MaxPooling_Step_Forward3D(img,Z,fH,fW,stride,Hlim,Wlim,Clim):

   """
   img -- (n_H_prev_pad,n_W_prev_pad,n_C)
   Z -- (n_H,n_W,n_C)
   """
   
   n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
   n_W = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
   n_C = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

   
   if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

      #set max_val and find max

      #Starting point of kernel
      IMG_H = n_H*stride 
      IMG_W = n_W*stride
      
      max_val = img[IMG_H,IMG_W,n_C]

      #Get the slice
      for h in range(fH):

         for w in range(fW):

            IMG_H = n_H*stride + h
            IMG_W = n_W*stride + w

            tmp = img[IMG_H,IMG_W,n_C]

            if tmp > max_val:

               max_val = tmp

      Z[n_H,n_W,n_C] = max_val
      
#Backward
def MaxPooling_Backward3D(dZ,cacheL,threadsperblock=(8,8,8)):

  """
  dZ -- (m,n_H,n_W,n_C)
  cacheL : (Z_prev,Z,kernel_size,stride,opadH,opadW)
  """
  #Get Cache
  Z_prev,Z,kernel_size,stride,opadH,opadW = cacheL

  #Get Z_prev_shape n_C_prev = n_C
  m,n_H_prev,n_W_prev,n_C_prev = Z_prev.shape

  #Get kernel size
  fH,fW = kernel_size

  #Padding
  if opadH == 0 and opadW == 0:

    Z_prev_pad = Z_prev.copy()
    dZ_prev_pad = np.zeros_like(Z_prev_pad)

  else:

    Z_prev_pad = zero_padding(Z_prev,opadH,opadW)
    dZ_prev_pad = np.zeros_like(Z_prev_pad)
    
  #Define stream
  number_of_streams = m
  stream_list = []

  for i in range(number_of_streams):

    stream_list.append(cuda.stream())

  #Define Blockspergrid n_C_prev = n_C
  m,n_H_prev_pad,n_W_prev_pad,n_C = dZ_prev_pad.shape
  
  blockspergrid_H = int(math.ceil(n_H_prev_pad/threadsperblock[0]))
  blockspergrid_W = int(math.ceil(n_W_prev_pad/threadsperblock[1]))
  blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

  blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

  #Backward
  for s in range(number_of_streams):

    #Move to device
    dZ_prev_pad_device = cuda.to_device((dZ_prev_pad[s,:,:,:]).copy(),stream=stream_list[s])
    Z_prev_pad_device = cuda.to_device((Z_prev_pad[s,:,:,:]).copy(),stream=stream_list[s])
    dZ_device = cuda.to_device((dZ[s,:,:,:]).copy(),stream=stream_list[s])
    Z_device = cuda.to_device((Z[s,:,:,:]).copy(),stream=stream_list[s])

    MaxPooling_Step_Backward3D[blockspergrid,threadsperblock,stream_list[s]](dZ_prev_pad_device,Z_prev_pad_device,dZ_device,Z_device,fH,fW,stride,n_H_prev_pad,n_W_prev_pad,n_C)
    cuda.synchronize()

    dZ_prev_pad[s,:,:,:] = dZ_prev_pad_device.copy_to_host(stream=stream_list[s])

  #Get result
  if opadH == 0 and opadW == 0:

    dZ_prev = dZ_prev_pad

  elif opadH != 0 and opadW != 0:

    dZ_prev = dZ_prev_pad[:,opadH:-opadH,opadW:-opadW,:]

  elif opadH == 0 and opadW != 0:

    dZ_prev = dZ_prev_pad[:,:,opadW:-opadW,:]

  elif opadH != 0 and opadW == 0:

    dZ_prev = dZ_prev_pad[:,opadH:-opadH,:,:]


  return dZ_prev

    
@cuda.jit("float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],int64,int64,int64,int64,int64,int64")
def MaxPooling_Step_Backward3D(dZ_prev,Z_prev,dZ,Z,fH,fW,stride,Hlim,Wlim,Clim):

  """
  Z_prev -- (n_H_prev_pad,n_W_prev_pad,n_C_prev)
  dZ_prev -- (n_H_prev_pad,n_W_prev_pad,n_C_prev)
  Z -- (n_H,n_W,n_C)
  dZ -- (n_H,n_W,n_C)
  """
  
  IMG_H = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  IMG_W = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  n_C = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (IMG_H < Hlim) and (IMG_W < Wlim) and (n_C < Clim):

    for h in range(fH):

      for w in range(fW):

        nh = (IMG_H-h)/stride
        nw = (IMG_W-w)/stride

        #check if nh,nw are non-negative integer
        if (nh >= 0) and (nw >= 0) and ((nh - int(nh)) == 0) and ((nw - int(nw)) == 0):

          #convert back to int as nh and nw become float after division
          nh = int(nh)
          nw = int(nw)

          #Check if it is the max value
          if Z_prev[IMG_H,IMG_W,n_C] == Z[nh,nw,n_C]:

            dZ_prev[IMG_H,IMG_W,n_C] = dZ_prev[IMG_H,IMG_W,n_C] + dZ[nh,nw,n_C]

          
#Average Pooling

#Forward
def AvgPooling_Forward3D(Z_prev,kernel_size,stride,threadsperblock=(8,8,8),padH=0,padW=0,padding = "Valid"):

   """
   Z_prev -- (m,n_H_prev,n_W_prev,n_C_prev)
   Kernel_size -- tuple(fH,fW)
   """

   m,n_H_prev,n_W_prev,n_C_prev = Z_prev.shape
   fH,fW = kernel_size

   #Padding
   if padH == 0 and padW == 0:

      Z_prev_pad = Z_prev.copy()

      n_H = int((n_H_prev-fH)/stride) + 1
      n_W = int((n_W_prev-fW)/stride) + 1
      n_C = n_C_prev

      opadH = 0
      opadW = 0

   elif padding == "Same":

        Z_prev_pad,opadH,opadW = same_padding(Z_prev,stride,fH,fW)

        n_H = n_H_prev
        n_W = n_W_prev
        n_C = n_C_prev
        
   elif padH != 0 or padW != 0:

      Z_prev_pad = zero_padding(Z_prev,padH,padW)

      n_H = int((n_H_prev+2*padH-fH)/stride) + 1
      n_W = int((n_W_prev+2*padW-fW)/stride) + 1
      n_C = n_C_prev

      opadH = padH
      opadW = padW

   #Set up
   Z = np.zeros((m,n_H,n_W,n_C))

   #Define Stream
   stream_list = []
   number_of_streams = m

   for i in range(number_of_streams):

      stream_list.append(cuda.stream())

   #Define Blocks per grid
   blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
   blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
   blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

   blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

   #MaxPooling Forward
   for s in range(number_of_streams):

      Z_prev_pad_device = cuda.to_device(Z_prev_pad[s,:,:,:].copy(),stream=stream_list[s])
      Z_device = cuda.to_device(Z[s,:,:,:].copy(),stream=stream_list[s])

      AvgPooling_Step_Forward3D[blockspergrid,threadsperblock,stream_list[s]](Z_prev_pad_device,Z_device,fH,fW,stride,n_H,n_W,n_C)
      cuda.synchronize()

      Z[s,:,:,:] = Z_device.copy_to_host(stream=stream_list[s])

   cacheL = (Z_prev,Z,kernel_size,stride,opadH,opadW)

   return Z,cacheL

@cuda.jit("float64[:,:,:],float64[:,:,:],int64,int64,int64,int64,int64,int64")
def AvgPooling_Step_Forward3D(img,Z,fH,fW,stride,Hlim,Wlim,Clim):

   """
   img -- (n_H_prev_pad,n_W_prev_pad,n_C)
   Z -- (n_H,n_W,n_C)
   """
   
   n_H = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
   n_W = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
   n_C = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

   
   if (n_H < Hlim) and (n_W < Wlim) and (n_C < Clim):

      #set sum and find mean
      sum_val = 0

      #Get the slice
      for h in range(fH):

         for w in range(fW):

            IMG_H = n_H*stride + h
            IMG_W = n_W*stride + w

            sum_val = sum_val + img[IMG_H,IMG_W,n_C]

      Z[n_H,n_W,n_C] = sum_val/(fH*fW)

#Backward
def AvgPooling_Backward3D(dZ,cacheL,threadsperblock=(8,8,8)):

  """
  dZ -- (m,n_H,n_W,n_C)
  cacheL : (Z_prev,Z,kernel_size,stride,opadH,opadW)
  """
  #Get Cache
  Z_prev,Z,kernel_size,stride,opadH,opadW = cacheL

  #Get Z_prev_shape n_C_prev = n_C
  m,n_H_prev,n_W_prev,n_C_prev = Z_prev.shape

  #Get kernel size
  fH,fW = kernel_size

  #Padding
  if opadH == 0 and opadW == 0:

    Z_prev_pad = Z_prev.copy()
    dZ_prev_pad = np.zeros_like(Z_prev_pad)

  else:

    Z_prev_pad = zero_padding(Z_prev,opadH,opadW)
    dZ_prev_pad = np.zeros_like(Z_prev_pad)
    
  #Define stream
  number_of_streams = m
  stream_list = []

  for i in range(number_of_streams):

    stream_list.append(cuda.stream())

  #Define Blockspergrid n_C_prev = n_C
  m,n_H_prev_pad,n_W_prev_pad,n_C = dZ_prev_pad.shape
  
  blockspergrid_H = int(math.ceil(n_H_prev_pad/threadsperblock[0]))
  blockspergrid_W = int(math.ceil(n_W_prev_pad/threadsperblock[1]))
  blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

  blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

  #Backward
  for s in range(number_of_streams):

    #Move to device
    dZ_prev_pad_device = cuda.to_device((dZ_prev_pad[s,:,:,:]).copy(),stream=stream_list[s])
    dZ_device = cuda.to_device((dZ[s,:,:,:]).copy(),stream=stream_list[s])

    AvgPooling_Step_Backward3D[blockspergrid,threadsperblock,stream_list[s]](dZ_prev_pad_device,dZ_device,fH,fW,stride,n_H_prev_pad,n_W_prev_pad,n_C)
    cuda.synchronize()

    dZ_prev_pad[s,:,:,:] = dZ_prev_pad_device.copy_to_host(stream=stream_list[s])

  #Get result
  if opadH == 0 and opadW == 0:

    dZ_prev = dZ_prev_pad

  elif opadH != 0 and opadW != 0:

    dZ_prev = dZ_prev_pad[:,opadH:-opadH,opadW:-opadW,:]

  elif opadH == 0 and opadW != 0:

    dZ_prev = dZ_prev_pad[:,:,opadW:-opadW,:]

  elif opadH != 0 and opadW == 0:

    dZ_prev = dZ_prev_pad[:,opadH:-opadH,:,:]


  return dZ_prev

@cuda.jit("float64[:,:,:],float64[:,:,:],int64,int64,int64,int64,int64,int64")
def AvgPooling_Step_Backward3D(dZ_prev,dZ,fH,fW,stride,Hlim,Wlim,Clim):

  """
  dZ_prev -- (n_H_prev_pad,n_W_prev_pad,n_C_prev)
  dZ -- (n_H,n_W,n_C)
  """
  
  IMG_H = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  IMG_W = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  n_C = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (IMG_H < Hlim) and (IMG_W < Wlim) and (n_C < Clim):


    n_H,n_W,_ = dZ.shape

    for h in range(fH):

      for w in range(fW):

        for nh in range(n_H):

          for nw in range(n_W):

            tmp_H = nh * stride + h
            tmp_W = nw * stride + w

            if (tmp_H == IMG_H) and (tmp_W == IMG_W):

              dZ_prev[IMG_H,IMG_W,n_C] = dZ_prev[IMG_H,IMG_W,n_C] + dZ[nh,nw,n_C]/(fH*fW)
    
    """
    for h in range(fH):

      for w in range(fW):

        nh = (IMG_H-h)/stride
        nw = (IMG_W-w)/stride

        #check if nh,nw are non-negative integer
        if (nh >= 0) and (nw >= 0) and ((nh - int(nh)) == 0) and ((nw - int(nw)) == 0):

          #convert back to int as nh and nw become float after division
          nh = int(nh)
          nw = int(nw)

          dZ_prev[IMG_H,IMG_W,n_C] = dZ_prev[IMG_H,IMG_W,n_C] + dZ[nh,nw,n_C]/(fH*fW)
    """
   

if __name__ == "__main__":

        A_device = cuda.to_device(np.zeros((3,3,1)).astype("float64"))
        B_device = cuda.to_device(np.ones((2,2,1)).astype("float64"))
        
        AvgPooling_Step_Backward3D[(1,1,1),(8,8,8)](A_device,B_device,2,2,1,3,3,1)
        cuda.synchronize()
        C = A_device.copy_to_host()
  
        
        #Test
        import Layers
        
        #Average Pooling Forward
      
        np.random.seed(1)
        """
        A_prev = np.random.randn(2, 5, 5, 3)
        stride,fH,fW = 1,3,3
        """
        """
        np.random.seed(1)
        A_prev = np.random.randn(2, 54, 96, 64)
        stride,fH,fW = 2,3,3
        
        obj = Layers.PoolingLayer()

        #GPU
        gpu_time = time.time()
        Z,cacheL = AvgPooling_Forward3D(A_prev,(fH,fW),stride,threadsperblock=(8,8,8))
        print(f"GPU: {time.time()-gpu_time}")
        print("Z.shape = " + str(Z.shape))
        print()

        #CPU
        cpu_time = time.time()
        A, cache = obj.Pooling_Forward(A_prev, stride,fH,fW, mode = "AVERAGE")
        print(f"CPU: {time.time()-cpu_time}")
        print("A.shape = " + str(A.shape))
        print()

        print(np.allclose(Z,A))
        """
        
        #Average pooling backward
        obj = Layers.PoolingLayer()
        """
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        fH,fW,stride = 2,2,1
        dA = np.random.randn(5, 4, 2, 2)
        """
        np.random.seed(1)
        A_prev = np.random.randn(2,100,100,64)
        stride,fH,fW = 2,3,3
        dA = np.random.randn(2,49,49, 64)

        #GPU
        Z,cacheL = AvgPooling_Forward3D(A_prev,(fH,fW),stride,threadsperblock=(8,8,8))
        
        gpu_time = time.time()
        dZ_prev = AvgPooling_Backward3D(dA,cacheL,threadsperblock=(8,8,8))
        print(f"GPU: {time.time()-gpu_time}")
        print()
        

        #CPU
        A, cachePL = obj.Pooling_Forward(A_prev,stride,fH,fW,mode = "AVERAGE")
        
        cpu_time = time.time()
        dA_prev = obj.Pooling_Backward(dA, cachePL, mode = "AVERAGE")
        print(f"CPU: {time.time()-cpu_time}")
        print()

        print(np.allclose(dZ_prev,dA_prev))
     
   
        #MaxPooling backward
        """
        np.random.seed(1)
       
        A_prev = np.random.randn(2,100,100,64)
        stride,fH,fW = 2,3,3
        dA = np.random.randn(2,49,49, 64)
        """
        """
        A_prev = np.random.randn(5, 5, 3, 2)
        fH,fW,stride = 2,2,1
        dA = np.random.randn(5, 4, 2, 2)
        """
        """
        obj = Layers.PoolingLayer()
        
        #GPU
        Z,cacheL = MaxPooling_Forward3D(A_prev,(fH,fW),stride,threadsperblock=(8,8,8))
        
        gpu_time = time.time()
        dZ_prev = MaxPooling_Backward3D(dA,cacheL,threadsperblock=(8,8,8))
        print(f"GPU: {time.time()-gpu_time}")
        print()
        
        #CPU
        A, cachePL = obj.Pooling_Forward(A_prev,stride,fH,fW)

        cpu_time = time.time()
        dA_prev = obj.Pooling_Backward(dA, cachePL, mode = "MAX")
        print(f"CPU: {time.time()-cpu_time}")
        print()

        print(np.allclose(dZ_prev,dA_prev))
        """
        
        #MaxPooling Forward
        """
        np.random.seed(1)

        A_prev = np.random.randn(2,600,600,64)
        stride,fH,fW = 2,3,3
        """
        
        """
        A_prev = np.random.randn(2, 5, 5, 3)
        stride,fH,fW = 1,3,3
        """

        """
        obj = Layers.PoolingLayer()
        
        #GPU
        gpu_time = time.time()
        Z,cacheL = MaxPooling_Forward3D(A_prev,(fH,fW),stride,threadsperblock=(8,8,8))
        print(f"GPU: {time.time()-gpu_time}")
        print("Z.shape = " + str(Z.shape))
        print()
        
        #CPU
        cpu_time = time.time()
        A, cache = obj.Pooling_Forward(A_prev, stride,fH,fW)
        print(f"CPU: {time.time()-cpu_time}")
        print("A.shape = " + str(A.shape))
        print()

        print(np.allclose(Z,A))
        """

        """
        #Convolution
        obj = Layers.ConvLayer()
        """
        #conv backward main function
        """
        np.random.seed(1)
        img = np.random.randn(10,4,4,3)
        W = np.random.randn(2,2,3,8)
        b = np.random.randn(1,1,1,8)
        stride = 2
        """
        """
        np.random.seed(1)
        img = np.random.randn(10,331,331,3)
        W = np.random.randn(3,3,3,64)
        b = np.random.randn(1,1,1,64)
        padH,padW,stride = 2,2,3
      
        
        Z,cacheL = Conv_Forward3D_GPU(img,W,b,stride,padH=2,padW=2,threadsperblock=(8,8,8))
        cuda.synchronize()

        m,n_H,n_W,n_C = Z.shape
        gpu_time = time.time()
        dA_prev,dW,db = Conv_backward3D_GPU(Z,cacheL,threadsperblock=(8,8,8))#obj.conv_backward(Z, cacheL,lambda x:x)
        cuda.synchronize()
        print(f"With GPU:{time.time()-gpu_time}")
        print("\n")
        print("dA_mean =", np.mean(dA_prev))
        print("dW_mean =", np.mean(dW))
        print("db_mean =", np.mean(db))
        print("\n")
        
        #CPU       
        Z, cache_conv = obj.conv_forward(img, W, b, stride,padH,padW)
        cpu_time = time.time()
        dAc, dWc, dbc = obj.conv_backward(Z, cache_conv,lambda x:x)
        print(f"With CPU:{time.time()-cpu_time}")
        print("\n")
        print("dA_mean =", np.mean(dAc))
        print("dW_mean =", np.mean(dWc))
        print("db_mean =", np.mean(dbc))
        print("\n")
        print(np.allclose(dA_prev,dAc))
        print(np.allclose(dW,dWc))
        print(np.allclose(db,dbc))
        """
        """
        #padding
        np.random.seed(1)
        x = np.random.randn(4, 3, 3, 2)
        x_pad = zero_padding(x,2,2)
        print ("x.shape =\n", x.shape)
        print ("x_pad.shape =\n", x_pad.shape)
        print ("x[1,1] =\n", x[1,1])
        print ("x_pad[1,1] =\n", x_pad[1,1])
        """
   
        
        #conv forward main function
        """
        #Test value
        np.random.seed(1)
        obj = Layers.ConvLayer()
        A_prev = np.random.randn(10,5,7,4)
        W = np.random.randn(3,3,4,8)
        b = np.random.randn(1,1,1,8)
        padH,padW,stride = 1,1,2
        
        Z, cache_conv = Conv_Forward3D_GPU(A_prev,W,b,stride,padH,padW,threadsperblock=(4,4,8))

        print("Z's mean =\n", np.mean(Z))
        print("Z[3,2,1] =\n", Z[3,2,1])
        print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])
        Z, cache_conv =  Conv_Forward3D_GPU(A_prev, W, b,stride,padding="Same")
        print("Shape of Z:{}".format(Z.shape[1:3]))
        """

        """
        #GPU
        W = np.random.randn(3,3,3,128)
        b = np.random.randn(1,1,1,128)
        img = np.random.randn(3,1080,1920,3).astype(np.float64)
        stride = 2
        gpu_time = time.time()
        Z,_ = Conv_Forward3D_GPU(img,W,b,stride,padH=0,padW=0,padding="Valid",threadsperblock=(4,4,32))
        cuda.synchronize()
        print(f"With GPU:{time.time()-gpu_time}")
        k1 = Z.copy()

        #CPU
        obj = Layers.ConvLayer()
        cpu_time = time.time()
        Z,_= obj.conv_forward(img,W,b,stride)
        print(f"With CPU:{time.time()-cpu_time}")
        k2 = Z.copy()

        print(np.array_equal(k1.round(6),k2.round(6)))
        print(np.allclose(k1,k2))
        """
        """
        #same padding
        np.random.seed(1)
        A_prev = np.random.randn(10,5,7,4)
        W = np.random.randn(2,5,4,8)
        b = np.random.randn(1,1,1,8)
        padH,padW,stride = 1,1,2

        #GPU
        gpu_time = time.time()
        Zg, cache_conv =  Conv_Forward3D_GPU(A_prev, W, b,stride,padding="Same")
        print(f"GPU: {time.time()-gpu_time}")
        print("Z's mean =\n", np.mean(Zg))
        print("Z[3,2,1] =\n", Zg[3,2,1])
        print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])
        print("Shape of Z:{}".format(Zg.shape[1:3]))
        

        #CPU
        cpu_time = time.time()
        Zc, cache_conv = obj.conv_forward(A_prev, W, b,stride,padding="Same")
        print(f"CPU: {time.time()-cpu_time}")
        print("Z's mean =\n", np.mean(Zc))
        print("Z[3,2,1] =\n", Zc[3,2,1])
        print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])
        print("Shape of Z:{}".format(Zc.shape[1:3]))
        
        print(np.allclose(Zg,Zc))
        """
        
        """
        #3D with sample
        #GPU
        W = np.random.randn(3,3,3,64).astype(np.float64)
        b = np.random.randn(1,1,1,64).astype(np.float64)
        img = np.random.randn(3,108,192,3).astype(np.float64)

        fH,fW,n_C_prev,n_C = W.shape
        m,n_H_prev,n_W_prev,n_C_prev = img.shape
        stride = 2

        n_H = int((n_H_prev-fH)/stride)+1
        n_W = int((n_W_prev-fW)/stride)+1

        Z = np.zeros((m,n_H,n_W,n_C))

        gpu_time = time.time()

        threadsperblock = (4,4,32)

        blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

        W_device = cuda.to_device(W)
        img_device = cuda.to_device(img)
        b_device = cuda.to_device(b)
        Z_device = cuda.to_device(Z)
        cuda.synchronize()

        conv_step_forward3D_with_sample[blockspergrid,threadsperblock](W_device,img_device,b_device,Z_device,stride,n_H,n_W,n_C)

        cuda.synchronize()
        k1 = Z_device.copy_to_host()
        cuda.synchronize()

        print(f"With GPU:{time.time()-gpu_time}")

        #CPU
        obj = Layers.ConvLayer()
        cpu_time = time.time()
        Z,_= obj.conv_forward(img,W,b,stride)            
        k2 = Z.copy()
        print(f"With CPU:{time.time()-cpu_time}")

        print(np.array_equal(k1.round(6),k2.round(6)))
        print(np.allclose(k1,k2))
        
        """

        """
        #3D
        #GPU
        W = np.random.randn(3,3,3,16).astype(np.float64)
        b = np.random.randn(1,1,1,16).astype(np.float64)
        Img = np.random.randn(1,1080,1920,3).astype(np.float64)

        m,n_H_prev,n_W_prev,n_C_prev = Img.shape

        fH,fW = W.shape[0],W.shape[1]

        stride = 2
        n_H = int((n_H_prev-fH)/stride)+1
        n_W = int((n_W_prev-fW)/stride)+1
        n_C = 16

        Z = np.zeros((n_H,n_W,16))

        threadsperblock = (8,8,2)

        blockspergrid_H = int(math.ceil(Z.shape[0]/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(Z.shape[1]/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(Z.shape[2]/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)


        W_device = cuda.to_device(W)
        Img_device = cuda.to_device(Img[0,:,:,:])
        Z_device = cuda.to_device(Z)
        b_device = cuda.to_device(b)

        cuda.synchronize()

        gpu_time = time.time()
        conv_step_forward3D[blockspergrid,threadsperblock](W_device,Img_device,b_device,Z_device,stride,n_H,n_W,n_C)
        cuda.synchronize()
        k1 = Z_device.copy_to_host()
        print(f"With GPU:{time.time()-gpu_time}")


        #CPU
        obj = Layers.ConvLayer()
        cpu_time = time.time()
        Z,_= obj.conv_forward(Img,W,b,stride)            
        k2 = Z[0,:,:,:]
        print(f"With CPU:{time.time()-cpu_time}")

        print(np.array_equal(k1.round(6),k2.round(6)))
        """

        """
        #2D
        #GPU
        W = np.random.randn(3,3,16).astype(np.float64)
        b = np.random.randn(1,1,16).astype(np.float64)
        Img = np.random.randn(1080,1920).astype(np.float64)
        #W = np.ones((2,2,16))
        #b = np.ones((1,1,16))
        #Img = np.ones((108,192))

        n_H_prev,n_W_prev = Img.shape

        fH,fW = W.shape[0],W.shape[1]

        stride = 2
        n_H = int((n_H_prev-fH)/stride)+1
        n_W = int((n_W_prev-fW)/stride)+1
        n_C = 1

        Z = np.zeros((n_H,n_W,n_C))

        threadsperblock = (8,8,2)

        blockspergrid_H = int(math.ceil(Z.shape[0]/threadsperblock[0]))
        blockspergrid_W = int(math.ceil(Z.shape[1]/threadsperblock[1]))
        blockspergrid_C = int(math.ceil(Z.shape[2]/threadsperblock[2]))

        blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)


        W_device = cuda.to_device(W)
        Img_device = cuda.to_device(Img)
        Z_device = cuda.to_device(Z)
        b_device = cuda.to_device(b)

        cuda.synchronize()

        gpu_time = time.time()
        conv_step_forward2D[blockspergrid,threadsperblock](W_device,Img_device,b_device,Z_device,stride,n_H,n_W,n_C)
        cuda.synchronize()
        k1 = Z_device.copy_to_host()
        print(f"With GPU:{time.time()-gpu_time}")

        #CPU
        obj = Layers.ConvLayer()
        cpu_time = time.time()
        #Get a sample
        #W = np.ones((2,2,1))
        #b = np.ones((1,1,1))
        a_prev_pad = Img.copy()
        Z = np.zeros((n_H,n_W,n_C))

        #Loop over vertical axis 
        for h in range(n_H):

         vert_start = h*stride
         vert_end = vert_start + fH

         #Loop over horizontal axis
         for w in range(n_W):

             hori_start = w*stride
             hori_end = hori_start + fW

             #Slice current sample
             a_slice_prev = a_prev_pad[vert_start:vert_end,hori_start:hori_end]

             #For each filter
             for c in range(n_C):

                 Wc = W[:,:,c]
                 bc = b[:,:,c]
                    
                 Z[h,w,c] =  np.sum(a_slice_prev*Wc)+float(bc)
                 
        k2 = Z.copy()
        print(f"With CPU:{time.time()-cpu_time}")

        print(np.array_equal(k1.round(6),k2.round(6)))
        """
        
