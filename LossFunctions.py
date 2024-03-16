import numpy as np

class Loss:
  def trainCrossEntropyLoss(y,yPred):
    val=-np.sum(y*np.log(yPred+1e-9))
    return val

  def trainMseLoss(y,yPred):
    return (np.sum((y-yPred)**2))/y.shape[0]

  def valCrossEntropyLoss(Model,w,b,x,y,wd,epsilon=1e-10):
    a,h,yPred=Model.forward_propagation(w,b,x)
    val=-np.sum(y*np.log(yPred+1e-9))
    return val