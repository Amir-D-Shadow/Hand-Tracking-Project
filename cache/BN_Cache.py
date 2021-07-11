#BatchNormalization

#Helper Functions

#sum
@cuda.jit("float64[:,:,:,:],float64[:,:,:],int64,int64,int64,int64")
def sum_axis0_3D(x,sum_x,m,Hlim,Wlim,Clim):

  """
  x -- (m,n_H,n_W,n_C)
  sum_x -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    #sum over each sample via different axis
    for i in range(m):

      sum_x[nh,nw,nc] = sum_x[nh,nw,nc] + x[i,nh,nw,nc]

#mean
@cuda.jit("float64[:,:,:,:],float64[:,:,:],int64,int64,int64,int64")     
def mean_axis0_3D(x,mean_x,m,Hlim,Wlim,Clim):

  """
  x -- (m,n_H,n_W,n_C)
  mu -- mean
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      mean_x[nh,nw,nc] = mean_x[nh,nw,nc] + x[i,nh,nw,nc]/m

#variance
@cuda.jit("float64[:,:,:,:],float64[:,:,:],float64[:,:,:],int64,int64,int64,int64")
def var_axis0_3D(x,var_x,mean_x,m,Hlim,Wlim,Clim):

  """
  x -- (m,n_H,n_W,n_C)
  var_x -- (n_H,n_W,n_C)
  mean_x -- (n_H,n_W,n_C)
  """
  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      var_x[nh,nw,nc] = var_x[nh,nw,nc] + (x[i,nh,nw,nc] - mean_x[nh,nw,nc])*(x[i,nh,nw,nc] - mean_x[nh,nw,nc])/m
      

#Z_norm
@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:],float64[:,:,:],float64,int64,int64,int64,int64")
def Z_norm_3D(x,z_norm,var_x,mean_x,epsilon,m,Hlim,Wlim,Clim):

  """
  x -- (m,n_H,n_W,n_C)
  z_norm -- (m,n_H,n_W,n_C)
  var_x -- (n_H,n_W,n_C)
  mean_x -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      z_norm[i,nh,nw,nc] = (x[i,nh,nw,nc] - mean_x[nh,nw,nc])/math.sqrt(var_x[nh,nw,nc]+epsilon)

#Z_S
@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:],float64[:,:,:],int64,int64,int64,int64")
def Z_S_3D(z_s,z_norm,gamma,beta,m,Hlim,Wlim,Clim):

  """
  z_s -- (m,n_H,n_W,n_C)
  z_norm -- (m,n_H,n_W,n_C)
  gamma -- (n_H,n_W,n_C)
  beta -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      z_s[i,nh,nw,nc] = gamma[nh,nw,nc] * z_norm[i,nh,nw,nc] + beta[nh,nw,nc]

#Exponentially Weighted Average
@cuda.jit("float64[:,:,:],float64[:,:,:],float64,int64,int64,int64")
def exp_weighted_avg_3D(running_x,x,momentum,Hlim,Wlim,Clim):

  """
  running_x -- (n_H,n_W,n_C)
  x -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    running_x[nh,nw,nc] = momentum * running_x[nh,nw,nc] + (1 - momentum) * x[nh,nw,nc]

# derivative variance
@cuda.jit("float64[:,:,:],float64[:,:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:,:],float64,int64,int64,int64,int64")
def dev_var_3D(dvar,z,mean_z,var_z,dz_norm,epsilon,m,Hlim,Wlim,Clim):

  """
  dvar -- (n_H,n_W,n_C)
  z -- (m,n_H,n_W,n_C)
  mean_z -- (n_H,n_W,n_C)
  var_z -- (n_H,n_W,n_C)
  dz_norm -- (m,n_H,n_W,n_C)
  """
  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z
  
  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    std_z = math.pow(var_z[nh,nw,nc]+epsilon,1.5)

    for i in range(m):

      dvar[nh,nw,nc] = dvar[nh,nw,nc] + (-(z[i,nh,nw,nc]-mean_z[nh,nw,nc])*dz_norm[i,nh,nw,nc]/(2*std_z))
  

# derivative mean
@cuda.jit("float64[:,:,:],float64[:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:],float64[:,:,:],float64,int64,int64,int64,int64")
def dev_mean_3D(dmean,dvar,dz_norm,z,mean_z,var_z,epsilon,m,Hlim,Wlim,Clim):

  """
  dmean -- (n_H,n_W,n_C)
  dvar -- (n_H,n_W,n_C)
  dz_norm -- (m,n_H,n_W,n_C)
  z -- (m,n_H,n_W,n_C)
  mean_z -- (n_H,n_W,n_C)
  var_z -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z
  
  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    std_z = math.sqrt(var_z[nh,nw,nc]+epsilon)

    for i in range(m):

      dmean[nh,nw,nc] = dmean[nh,nw,nc] + (-dz_norm[i,nh,nw,nc]/std_z) + (-2*(z[i,nh,nw,nc]-mean_z[nh,nw,nc])*dvar[nh,nw,nc]/m) 

# derivative dz
@cuda.jit("float64[:,:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:],float64[:,:,:],float64,int64,int64,int64,int64")
def dev_z_3D(dz,dmean,dvar,dz_norm,z,mean_z,var_z,epsilon,m,Hlim,Wlim,Clim):

  """
  dz -- (m,n_H,n_W,n_C)
  dmean -- (n_H,n_W,n_C)
  dvar -- (n_H,n_W,n_C)
  dz_norm -- (m,n_H,n_W,n_C)
  z -- (m,n_H,n_W,n_C)
  mean_z -- (n_H,n_W,n_C)
  var_z -- (n_H,n_W,n_C)
  """
  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    std_z = math.sqrt(var_z[nh,nw,nc]+epsilon)

    for i in range(m):

      dz[i,nh,nw,nc] = dmean[nh,nw,nc]/m + 2*(z[i,nh,nw,nc]-mean_z[nh,nw,nc])*dvar[nh,nw,nc]/m + dz_norm[i,nh,nw,nc]/std_z

# derivative dgamma
@cuda.jit("float64[:,:,:],float64[:,:,:,:],float64[:,:,:,:],int64,int64,int64,int64")
def dev_gamma_3D(dgamma,dz_s,z_norm,m,Hlim,Wlim,Clim):

  """
  dgamma -- (n_H,n_W,n_C)
  dz_s -- (m,n_H,n_W,n_C)
  z_norm -- (m,n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      dgamma[nh,nw,nc] = dgamma[nh,nw,nc] + z_norm[i,nh,nw,nc] * dz_s[i,nh,nw,nc]

#derivative Z_NORM
@cuda.jit("float64[:,:,:,:],float64[:,:,:,:],float64[:,:,:],int64,int64,int64,int64")
def dev_z_norm(dz_norm,dz_s,gamma,m,Hlim,Wlim,Clim):

  """
  dz_norm -- (m,n_H,n_W,n_C)
  dz_s -- (m,n_H,n_W,n_C)
  gamma -- (n_H,n_W,n_C)
  """

  nh = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  nw = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
  nc = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

  if (nh < Hlim) and (nw < Wlim) and (nc < Clim):

    for i in range(m):

      dz_norm[i,nh,nw,nc] = gamma[nh,nw,nc] * dz_s[i,nh,nw,nc]
  

#BN Forward
def BatchNormalization_Forward3D(Z,batch_para,running_para,epsilon = 1e-10,threadsperblock = (8,8,8),mode="TRAIN"):

  """
  Z -- (m,n_H,n_W,n_C)
  running_para -- {running_mean,running_var,momentum}
  batch_para -- {gamma,beta}
  gamma -- (n_H,n_W,n_C)
  beta -- (n_H,n_W,n_C)
  """

  #set up
  running_mean = running_para["running_mean"]
  running_var = running_para["running_var"]
  momentum = running_para["momentum"]

  gamma = batch_para["gamma"]
  beta = batch_para["beta"]

  #Get shape Z
  m,n_H,n_W,n_C = Z.shape
  Z_device = cuda.to_device(Z)

  #blockspergrid
  blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
  blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
  blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

  blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

  #Forward
  if mode == "TRAIN":

    #calculate mean
    mean_x = np.zeros((n_H,n_W,n_C))
    mean_x_device = cuda.to_device(mean_x)
    mean_axis0_3D[blockspergrid,threadsperblock](Z_device,mean_x_device,m,n_H,n_W,n_C)
    cuda.synchronize()
    mean_x = mean_x_device.copy_to_host()

    #calculate var
    var_x = np.zeros((n_H,n_W,n_C))
    var_x_device = cuda.to_device(var_x)
    var_axis0_3D[blockspergrid,threadsperblock](Z_device,var_x_device,mean_x_device,m,n_H,n_W,n_C)
    cuda.synchronize()
    var_x = var_x_device.copy_to_host()
      
    #update running mean
    running_mean_device = cuda.to_device(running_mean)
    exp_weighted_avg_3D[blockspergrid,threadsperblock](running_mean_device,mean_x_device,momentum,n_H,n_W,n_C)
    cuda.synchronize()
    running_para["running_mean"] = running_mean_device.copy_to_host()

    #update running var
    running_var_device = cuda.to_device(running_var)
    exp_weighted_avg_3D[blockspergrid,threadsperblock](running_var_device,var_x_device,momentum,n_H,n_W,n_C)
    cuda.synchronize()
    running_para["running_var"] = running_var_device.copy_to_host()
    
    #Calculate Z_norm
    Z_NORM = np.zeros_like(Z)
    Z_NORM_device = cuda.to_device(Z_NORM)
    Z_norm_3D[blockspergrid,threadsperblock](Z_device,Z_NORM_device,var_x_device,mean_x_device,epsilon,m,n_H,n_W,n_C)
    cuda.synchronize()
    Z_NORM = Z_NORM_device.copy_to_host()

    #Calculate Z_S
    Z_S = np.zeros_like(Z)
    Z_S_device = cuda.to_device(Z_S)
    Z_S_3D[blockspergrid,threadsperblock](Z_S_device,Z_NORM_device,gamma,beta,m,n_H,n_W,n_C)
    cuda.synchronize()
    Z_S = Z_S_device.copy_to_host()

    #Save cache for backpropagation
    cacheL = (Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_x,var_x)

  else:

    #calculate Z_norm
    running_mean_device = cuda.to_device(running_mean)
    running_var_device = cuda.to_device(running_var)

    Z_NORM = np.zeros_like(Z)
    Z_NORM_device = cuda.to_device(Z_NORM)
    Z_norm_3D[blockspergrid,threadsperblock](Z_device,Z_NORM_device,running_var_x_device,running_mean_x_device,epsilon,m,n_H,n_W,n_C)
    cuda.synchronize()

    #Calculate Z_S
    Z_S = np.zeros_like(Z)
    Z_S_device = cuda.to_device(Z_S)
    Z_S_3D[blockspergrid,threadsperblock](Z_S_device,Z_NORM_device,gamma,beta,m,n_H,n_W,n_C)
    cuda.synchronize()
    Z_S = Z_S_device.copy_to_host()

    cacheL = ()

  return Z_S,cacheL

#BN Backward
def BatchNormalization_Backward3D(dZ_S,cacheL,threadsperblock=(8,8,8)):

  """
  dZ_S -- (m,n_H,n_W,n_C)
  cacheL -- (Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_x,var_x)
  gamma -- (n_H,n_W,n_C)
  beta -- (n_H,n_W,n_C)
  """
  #Get cache
  Z,Z_NORM,Z_S,gamma,beta,epsilon,mean_x,var_x = cacheL

  #Move gamma and beta to GPU
  gamma_device = cuda.to_device(gamma)
  beta_device = cuda.to_device(beta)

  #Move Z,Z_S,Z_NORM to GPU
  Z_S_device = cuda.to_device(Z_S)
  Z_NORM_device = cuda.to_device(Z_NORM)
  Z_device = cuda.to_device(Z)

  #Move mean_x,var_x to gpu
  mean_x_device = cuda.to_device(mean_x)
  var_x_device = cuda.to_device(var_x)
  
  #Get shape dZ_S
  m,n_H,n_W,n_C = dZ_S.shape
  dZ_S_device = cuda.to_device(dZ_S)

  #Define Blockspergrid
  blockspergrid_H = int(math.ceil(n_H/threadsperblock[0]))
  blockspergrid_W = int(math.ceil(n_W/threadsperblock[1]))
  blockspergrid_C = int(math.ceil(n_C/threadsperblock[2]))

  blockspergrid = (blockspergrid_H,blockspergrid_W,blockspergrid_C)

  #dZ_NORM
  dZ_NORM = np.zeros_like(dZ_S)
  dZ_NORM_device = cuda.to_device(dZ_NORM)
  dev_z_norm[blockspergrid,threadsperblock](dZ_NORM_device,dZ_S_device,gamma_device,m,n_H,n_W,n_C)
  cuda.synchronize()
    
  #dbeta
  dbeta = np.zeros_like(beta)
  dbeta_device = cuda.to_device(dbeta)
  sum_axis0_3D[blockspergrid,threadsperblock](dZ_S_device,dbeta_device,m,n_H,n_W,n_C)
  cuda.synchronize()
  dbeta = dbeta_device.copy_to_host()

  #dgamma
  dgamma = np.zeros_like(gamma)
  dgamma_device = cuda.to_device(dgamma)
  dev_gamma_3D[blockspergrid,threadsperblock](dgamma_device,dZ_S_device,Z_NORM_device,m,n_H,n_W,n_C)
  cuda.synchronize()
  dgamma = dgamma_device.copy_to_host()

  #dvar
  dvar = np.zeros_like(var_x)
  dvar_device = cuda.to_device(dvar)
  dev_var_3D[blockspergrid,threadsperblock](dvar_device,Z_device,mean_x_device,var_x_device,dZ_NORM_device,epsilon,m,n_H,n_W,n_C)
  cuda.synchronize()
  dvar = dvar_device.copy_to_host()

  #dmean
  dmean = np.zeros_like(mean_x)
  dmean_device = cuda.to_device(dmean)
  dev_mean_3D[blockspergrid,threadsperblock](dmean_device,dvar_device,dZ_NORM_device,Z_device,mean_x_device,var_x_device,epsilon,m,n_H,n_W,n_C)
  cuda.synchronize()
  dmean = dmean_device.copy_to_host()

  #dZ
  dZ = np.zeros_like(Z)
  dZ_device = cuda.to_device(dZ)
  dev_z_3D[blockspergrid,threadsperblock](dZ_device,dmean_device,dvar_device,dZ_NORM_device,Z_device,mean_x_device,var_x_device,epsilon,m,n_H,n_W,n_C)
  cuda.synchronize()
  dZ = dZ_device.copy_to_host()

  return dZ,dgamma,dbeta
