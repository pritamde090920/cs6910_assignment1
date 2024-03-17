import numpy as np

class Accuracy:
  def trainAccuracy(yTrue,y_pred):
    '''
      Parameteres:
        yTrue : True predictions in one hot vector form
        y_pred : Predcictions of the input in form of probabilitites (across 10 output classes returned by softmax)
      Returns:
        accuracy precentage of correct predictions
      Function:
        Performs accuracy calculation of training data
    '''
    accuracy=0
    for i in range(y_pred.shape[0]):
      '''checking if the true class value and the predicted class are same or not'''
      if(np.argmax(yTrue[i])==np.argmax(y_pred[i])):
        accuracy+=1

    return accuracy/yTrue.shape[0]

  def valAccuracy(Model,weights,biases,input,yTrue):
    '''
      Parameteres:
        Model : Object of neural network
        weights : weight matrix
        biases : bias matrix
        input : Input validation data in flattened (784x1) vector form
        yTrue : Output in one hot vector form
      Returns:
        accuracy precentage of correct predictions
      Function:
        Performs accuracy calculation of validation data
    '''
    accuracy=0
    '''doing forward prop to find out the predictions of the input according to the weight and bias'''
    a,h,y_pred=Model.forward_propagation(weights,biases,input)
    for i in range(y_pred.shape[0]):
      '''checking if the true class value and the predicted class are same or not'''
      if(np.argmax(yTrue[i])==np.argmax(y_pred)):
        accuracy+=1

    return accuracy/input.shape[0]

  def testAccuracy(yTrue,y_pred):
    '''
      Parameteres:
        yTrue : True predictions in one hot vector form
        y_pred : Predcictions of the input in form of probabilitites (across 10 output classes returned by softmax)
      Returns:
        accuracy precentage of correct predictions
      Function:
        Performs accuracy calculation of test data
    '''
    accuracy=0
    for i in range(yTrue.shape[0]):
      '''checking if the true class value and the predicted class are same or not'''
      if(np.argmax(yTrue[i])==np.argmax(y_pred[i])):
        accuracy+=1

    return accuracy/yTrue.shape[0]