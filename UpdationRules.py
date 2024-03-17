import numpy as np

class UpdateParameters:

  def update_parameters(weights,biases,eta,del_w,del_b,weight_decay):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
        eta : learning rate
        del_w : change in weight (matrix of matrices corresponding to each layer)
        del_b : change in bias (matrix of vectors corresponding to each layer)
        weight_decay : weight decay parameter
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Updates weights and biases according to the update rule
    '''
    for i in range(1,len(del_w)+1):
      weights[i]-=(eta*del_w[i]+eta*weight_decay*weights[i])
    for i in range(1,len(del_b)+1):
      biases[i]-=eta*del_b[i]
    return weights,biases


  def update_parameters_mgd(weights,biases,del_w,del_b,eta,weight_decay):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
        eta : learning rate
        del_w : change in weight (matrix of matrices corresponding to each layer)
        del_b : change in bias (matrix of vectors corresponding to each layer)
        weight_decay : weight decay parameter
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Updates weights and biases according to the update rule
    '''
    for i in range(1,len(del_w)+1):
      weights[i]-=(del_w[i]+eta*weight_decay*weights[i])
    for i in range(1,len(del_b)+1):
      biases[i]-=del_b[i]
    return weights,biases


  def update_parameters_rms(weights,biases,eta,vw,vb,del_w,del_b,eps,weight_decay):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
        eta : learning rate
        del_w : change in weight (matrix of matrices corresponding to each layer)
        del_b : change in bias (matrix of vectors corresponding to each layer)
        weight_decay : weight decay parameter
        vw : parameter to update the learning rate for weight
        vb : parameter to update the learning rate for bias
        eps : correction parameter for learning rate updation
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Updates weights and biases according to the update rule
    '''
    for i in range(1,len(vw)+1):
      updated_eta=eta/(np.sqrt(np.sum(vw[i]))+eps)
      weights[i]-=(updated_eta*del_w[i]+eta*weight_decay*weights[i])
    for i in range(1,len(vb)+1):
      updated_eta=eta/(np.sqrt(np.sum(vb[i]))+eps)
      biases[i]-=updated_eta*del_b[i]
    return weights,biases


  def update_parameters_adam(weights,biases,eta,mw_hat,mb_hat,vw_hat,vb_hat,eps,weight_decay):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
        eta : learning rate
        mw_hat : sclaed weight
        mb_hat : scaled bias
        vw_hat : parameter to update the learning rate for weight
        vb_hat : parameter to update the learning rate for bias
        eps : correction parameter for learning rate updation
        weight_decay : weight decay parameter
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Updates weights and biases according to the update rule
    '''
    for i in range(1,len(vw_hat)+1):
      weights[i]-=((eta*mw_hat[i]/(np.sqrt(vw_hat[i])+eps))+eta*weight_decay*weights[i])
    for i in range(1,len(vb_hat)+1):
      biases[i]-=(eta*mb_hat[i]/(np.sqrt(vb_hat[i])+eps))
    return weights,biases


  def update_parameters_nadam(weights,biases,eta,mw_hat,mb_hat,vw_hat,vb_hat,beta1,beta2,del_w,del_b,eps,weight_decay):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
        eta : learning rate
        mw_hat : sclaed weight
        mb_hat : scaled bias
        vw_hat : parameter to update the learning rate for weight
        vb_hat : parameter to update the learning rate for bias
        beta 1, beta 2 : parameters for updation
        del_w : change in weight (matrix of matrices corresponding to each layer)
        eps : correction parameter for learning rate updation
        del_b : change in bias (matrix of vectors corresponding to each layer)
        weight_decay : weight decay parameter
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Updates weights and biases according to the update rule
    '''
    for i in range(1,len(vw_hat)+1):
      weights[i]-=(((eta/(np.sqrt(vw_hat[i]+eps)))*((beta1*mw_hat[i])+(((1-beta1)*del_w[i])/(1-beta1**(i+1)))))+eta*weight_decay*weights[i])
    for i in range(1,len(vb_hat)+1):
      biases[i]-=(eta/(np.sqrt(vb_hat[i]+eps)))*((beta1*mb_hat[i])+(((1-beta1)*del_b[i])/(1-beta1**(i+1))))
    return weights,biases

  def normalizeParameters(weights,biases):
    '''
      Parameters:
        weights : weight matrix (matrix of all weights of each level which are individually a matrix)
        biases : bias matrix (matrix of all biases of each level which are individually a vector)
      Returns :
        two matrices (one for updated weight and one for updated bias)
      Function:
        Normalizes the parameteres
    '''
    for i in range(1,len(weights)+1):
      val=np.linalg.norm(weights[i])
      if(val>1.0):
        weights[i]=weights[i]/val
    for i in range(1,len(biases)+1):
      val=np.linalg.norm(biases[i])
      if(val>1.0):
        biases[i]=biases[i]/val
    return weights,biases