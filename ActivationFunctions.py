import numpy as np

class Activation:
  
  def sigmoid(x):
    '''
      Parameteres:
        x : input matrix of data
      Returns:
        matrix of same dimension as input
      Function:
        Performs sigmoid function operation on input matrix
    '''
    '''clipping the value netween -150 and 150 so that overflow does not occur'''
    clipped_x=np.clip(x,-150,150)
    '''performing sigmoid'''
    return 1/(1+np.exp(-clipped_x))

  def grad_sigmoid(x):
    '''
      Parameters:
        x : input matrix of data
      Returns :
        matrix of same dimension as input
      Function:
        Calculates gradient of sigmoid fucntion and applies it on input matrix
    '''
    '''clipping the value netween -150 and 150 so that overflow does not occur'''
    clipped_x=np.clip(x,-150,150)
    '''performing gradient of sigmoid'''
    s=1/(1+np.exp(-clipped_x))
    return s*(1-s)

  def tanh(x):
    '''
      Parameteres:
        x : input matrix of data
      Returns:
        matrix of same dimension as input
      Function:
        Performs tanh function operation on input matrix
    '''
    return np.tanh(x)

  def grad_tanh(x):
    '''
      Parameters:
        x : input matrix of data
      Returns :
        matrix of same dimension as input
      Function:
        Calculates gradient of tanh fucntion and applies it on input matrix
    '''
    return 1-(np.tanh(x)**2)

  def relu(x):
    '''
      Parameteres:
        x : input matrix of data
      Returns:
        matrix of same dimension as input
      Function:
        Performs relu function operation on input matrix
    '''
    return np.maximum(x,0)

  def grad_relu(x):
    '''
      Parameters:
        x : input matrix of data
      Returns :
        matrix of same dimension as input
      Function:
        Calculates gradient of relu fucntion and applies it on input matrix
    '''
    return 1*(x>0)
  
  def identity(x):
    '''
      Parameteres:
        x : input matrix of data
      Returns:
        matrix of same dimension as input
      Function:
        Performs identity function operation on input matrix
    '''
    return x
  
  def grad_identity(x):
    '''
      Parameters:
        x : input matrix of data
      Returns :
        matrix of same dimension as input
      Function:
        Calculates gradient of identity fucntion and applies it on input matrix
    '''
    a,b=x.shape
    return np.ones((a,b))

  def softmax(a):
    '''
      Parameteres:
        x : input matrix of data
      Returns:
        matrix of same dimension as input
      Function:
        Performs softmax function operation on inout matrix
    '''
    '''normalizing the value so that overflow does not occur'''
    exp_a=np.exp(a-np.max(a))
    '''performing softmax'''
    sum_exp=np.sum(exp_a)
    return exp_a/sum_exp

  def grad_softmax(self,a):
    '''
      Parameters:
        x : input matrix of data
      Returns :
        matrix of same dimension as input
      Function:
        Calculates gradient of softmax fucntion and applies it on input matrix
    '''
    return self.softmax(a)*(1-self.softmax(a))