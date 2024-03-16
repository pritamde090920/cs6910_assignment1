import numpy as np

class Accuracy:
  def trainAccuracy(y,y_pred):
    accuracy=0
    for i in range(y_pred.shape[0]):
      if(np.argmax(y[i])==np.argmax(y_pred[i])):
        accuracy+=1

    return accuracy/y.shape[0]

  def valAccuracy(Model,w,b,x,y):
    accuracy=0
    a,h,y_pred=Model.forward_propagation(w,b,x)
    for i in range(y_pred.shape[0]):
      if(np.argmax(y[i])==np.argmax(y_pred)):
        accuracy+=1

    return accuracy/x.shape[0]

  def testAccuracy(y,y_pred):
    accuracy=0
    for i in range(y.shape[0]):
      if(np.argmax(y[i])==np.argmax(y_pred[i])):
        accuracy+=1

    return accuracy/y.shape[0]