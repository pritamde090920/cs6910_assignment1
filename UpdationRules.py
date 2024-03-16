import numpy as np

class UpdateParameters:
  def update_parameters(W,B,eta,del_w,del_b,wd):
    for i in range(1,len(del_w)+1):
      W[i]-=(eta*del_w[i]+eta*wd*W[i])
    for i in range(1,len(del_b)+1):
      B[i]-=eta*del_b[i]
    return W,B

  def update_parameters_mgd(W,B,del_w,del_b,eta,wd):
    for i in range(1,len(del_w)+1):
      W[i]-=(del_w[i]+eta*wd*W[i])
    for i in range(1,len(del_b)+1):
      B[i]-=del_b[i]
    return W,B

  def update_parameters_rms(W,B,eta,vw,vb,del_w,del_b,eps,wd):
    for i in range(1,len(vw)+1):
      updated_eta=eta/(np.sqrt(np.sum(vw[i]))+eps)
      W[i]-=(updated_eta*del_w[i]+eta*wd*W[i])
    for i in range(1,len(vb)+1):
      updated_eta=eta/(np.sqrt(np.sum(vb[i]))+eps)
      B[i]-=updated_eta*del_b[i]
    return W,B

  def update_parameters_adam(W,B,eta,mw_hat,mb_hat,vw_hat,vb_hat,eps,wd):
    for i in range(1,len(vw_hat)+1):
      # updated_eta=eta/(np.sqrt(vw_hat[i]+eps))
      W[i]-=((eta*mw_hat[i]/(np.sqrt(vw_hat[i])+eps))+eta*wd*W[i])
    for i in range(1,len(vb_hat)+1):
      # updated_eta=eta/(np.sqrt(vb_hat[i]+eps))
      B[i]-=(eta*mb_hat[i]/(np.sqrt(vb_hat[i])+eps))
    return W,B

  def update_parameters_nadam(W,B,eta,mw_hat,mb_hat,vw_hat,vb_hat,beta1,beta2,del_w,del_b,eps,wd):
    for i in range(1,len(vw_hat)+1):
      W[i]-=(((eta/(np.sqrt(vw_hat[i]+eps)))*((beta1*mw_hat[i])+(((1-beta1)*del_w[i])/(1-beta1**(i+1)))))+eta*wd*W[i])
    for i in range(1,len(vb_hat)+1):
      B[i]-=(eta/(np.sqrt(vb_hat[i]+eps)))*((beta1*mb_hat[i])+(((1-beta1)*del_b[i])/(1-beta1**(i+1))))
    return W,B

  def normalizeParameters(W,B):
    for i in range(1,len(W)+1):
      val=np.linalg.norm(W[i])
      if(val>1.0):
        W[i]=W[i]/val
    for i in range(1,len(B)+1):
      val=np.linalg.norm(B[i])
      if(val>1.0):
        B[i]=B[i]/val
    return W,B