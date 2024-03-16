import numpy as np

class Activation:
  def sigmoid(x):
    clipped_x=np.clip(x,-150,150)
    return 1/(1+np.exp(-clipped_x))

  def grad_sigmoid(x):
    clipped_x=np.clip(x,-150,150)
    s=1/(1+np.exp(-clipped_x))
    return s*(1-s)

  def tanh(x):
    return np.tanh(x)

  def grad_tanh(x):
    return 1-(np.tanh(x)**2)

  def relu(x):
    return np.maximum(x,0)

  def grad_relu(x):
    return 1*(x>0)

  def softmax(a):
    exp_a=np.exp(a-np.max(a))
    sum_exp=np.sum(exp_a)
    return exp_a/sum_exp

  def grad_softmax(self,a):
    return self.softmax(a)*(1-self.softmax(a))