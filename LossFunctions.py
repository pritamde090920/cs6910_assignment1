import numpy as np

class Loss:

  def trainCrossEntropyLoss(yTrue,yPred):
    '''
      Parameters:
        yTrue : True predictions in one hot vector form
        y_pred : Predcictions of the input in form of probabilitites (across 10 output classes returned by softmax)
      Returns :
        loss value
      Function:
        Calculates cross entopy loss on input
    '''
    val=-np.sum(yTrue*np.log(yPred+1e-9)) #softmax function
    return val

  def trainMseLoss(yTrue,yPred):
    '''
      Parameters:
        yTrue : True predictions in one hot vector form
        y_pred : Predcictions of the input in form of probabilitites (across 10 output classes returned by softmax)
      Returns :
        loss value
      Function:
        Calculates mean squared loss on input
    '''
    return (np.sum((yTrue-yPred)**2))/yTrue.shape[0]