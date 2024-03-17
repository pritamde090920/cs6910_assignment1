'''
    This file is created to supprt the train.py implementation.
    The file NeuralNetowk.py supports all the specific questions of the assignment.
    The difference between the two files is that in this file, the dataset is not read as the dataset can be either fashion_mnist or mnist. 
    Whereas in the other file, the fashion_mnist dataset is always read.
'''


import wandb
import numpy as np
import math
from keras.datasets import fashion_mnist,mnist
from ParameterInitialization import ParamaterInitialization
from ActivationFunctions import Activation
from UpdationRules import UpdateParameters
from AccuracyCalculation import Accuracy
from LossFunctions import Loss


class FeedForwardNeuralNetwork:

  '''class variables'''
  epochs=0
  hls=0
  neurons_in_hl=0
  eta=0.0
  activation=0
  num_of_layers=0
  x_train,x_val,x_test=np.zeros(784),np.zeros(784),np.zeros(784)
  y_train,y_val,y_test=np.zeros(10),np.zeros(10),np.zeros(10)
  W=dict()
  B=dict()
  wd=0.0
  mnistRun=False
  loss=''

  '''constructor to set all the class parameters'''
  def __init__ (self,x_train1,y_train1,x_val1,y_val1,x_test1,y_test1,hls,neurons_in_hl,activation,initialization,epochs,eta,wd,loss):
    '''In this constructor, the train, validation and tets data are passed. the dataset is read in the train.py file based ont he type of dataset'''

    '''flattening the input and one hot encoding the output'''
    self.x_train=self.input_flattening(x_train1)
    self.y_train=self.one_hot_encoding(y_train1)

    self.x_val=self.input_flattening(x_val1)
    self.y_val=self.one_hot_encoding(y_val1)

    self.x_test=self.input_flattening(x_test1)
    self.y_test=self.one_hot_encoding(y_test1)

    self.epochs=epochs
    self.hls=hls
    self.neurons_in_hl=neurons_in_hl
    self.eta=eta
    self.activation=activation
    self.num_of_layers=hls+2
    self.initialization=initialization
    self.wd=wd
    self.loss=loss

    '''initializing the weights and biases'''
    self.W=ParamaterInitialization.initializeW(self.initialization,self.num_of_layers,self.neurons_in_hl,self.x_train.shape[1],self.y_train.shape[1])
    self.B=ParamaterInitialization.initializeB(self.initialization,self.num_of_layers,self.neurons_in_hl,self.x_train.shape[1],self.y_train.shape[1])


  def one_hot_encoding(self,y):
    '''
      Parameters:
        y : output
      Returns :
        a vector
      Function:
        creates a one hot vector
    '''
    temp=list()
    for i in range(y.shape[0]):
      vector=np.zeros(10)
      vector[y[i]]=1
      temp.append(vector)
    return np.array(temp)

  def input_flattening(self,x):
    '''
      Parameters:
        x : input
      Returns :
        a vector
      Function:
        performs flattening of the input
    '''
    return x.reshape(x.shape[0],-1)/255.0

  def initGrads(self):
    '''
      Parameters:
        None
      Returns :
        two matrices (one for weight and one for bias)
      Function:
        initializes the update matrices of the weight and bias
    '''
    del_w=dict()
    del_w[1]=np.zeros((self.neurons_in_hl,self.x_train.shape[1]), dtype=np.float64)
    for i in range(2,self.num_of_layers-1):
      del_w[i]=np.zeros((self.neurons_in_hl,self.neurons_in_hl), dtype=np.float64)
    del_w[self.num_of_layers-1]=np.zeros((self.y_train.shape[1],self.neurons_in_hl), dtype=np.float64)

    del_b=dict()
    for i in range(1,self.num_of_layers-1):
      del_b[i]=np.zeros(self.neurons_in_hl, dtype=np.float64)
    del_b[self.num_of_layers-1]=np.zeros(self.y_train.shape[1], dtype=np.float64)

    return del_w,del_b


  def forward_propagation_test(self,w,b,x):
    '''
      Parameters:
        w : weights
        b : biases
        x : input
      Returns :
        a : pre activation
        h : activation
        y_cap : prediction of probabilities over the 10 output classes
      Function:
        performs forward propagation. this is a utility function to perform forward propagation to get output probabilities for calculating accuracies
    '''
    h=dict()
    a=dict()
    '''input activation is same as input'''
    h[0]=x

    '''perform activation functions from input to pre output layer'''
    for k in range(1,self.num_of_layers-1):
      a[k]=b[k]+np.dot(w[k],h[k-1])
      if(self.activation=="sigmoid"):
        h[k]=Activation.sigmoid(a[k])
      elif(self.activation=="tanh"):
        h[k]=Activation.tanh(a[k])
      elif(self.activation=="identity"):
        h[k]=Activation.identity(a[k])
      else:
        h[k]=Activation.relu(a[k])

    '''perform softmax fucntion in the output layer'''
    a[self.num_of_layers-1]=b[self.num_of_layers-1]+np.dot(w[self.num_of_layers-1],h[self.num_of_layers-2])
    y_cap=Activation.softmax(a[self.hls+1])

    return a,h,y_cap

  def forward_propagation(self,w,b,x):
    '''
      Parameters:
        w : weights
        b : biases
        x : input
      Returns :
        a : pre activation
        h : activation
        y_cap : prediction of probabilities over the 10 output classes
      Function:
        performs forward propagation
    '''
    h=dict()
    a=dict()
    b1=dict()

    h[0]=x

    for k in range(1,self.num_of_layers-1):
      b1[k]=b[k].reshape(1,-1)
      b1[k]=np.repeat(b1[k],x.shape[1],axis=0).transpose()
      a[k]=b1[k]+np.matmul(w[k],h[k-1])
      if(self.activation=="sigmoid"):
        h[k]=Activation.sigmoid(a[k])
      elif(self.activation=="tanh"):
        h[k]=Activation.tanh(a[k])
      elif(self.activation=="identity"):
        h[k]=Activation.identity(a[k])
      else:
        h[k]=Activation.relu(a[k])

    b1[self.num_of_layers-1]=b[self.num_of_layers-1].reshape(1,-1)
    b1[self.num_of_layers-1]=np.repeat(b1[self.num_of_layers-1],x.shape[1],axis=0).transpose()
    a[self.num_of_layers-1]=b1[self.num_of_layers-1]+np.matmul(w[self.num_of_layers-1],h[self.num_of_layers-2])
    temp=a[self.num_of_layers-1].transpose()
    l=list()
    for i in range(len(temp)):
      l.append(Activation.softmax(temp[i]))
    y_cap=np.array(l)

    return a,h,y_cap.T

  def backward_propagation(self,h,a,y,y_cap):
    '''
      Parameters:
        h : activation
        a : pre activation
        y : true output
        y_cap : predicted output
      Returns :
        del_w : update in weights
        del_b : update in biases
      Function:
        performs backward propagation
    '''
    del_a=dict()
    del_w=dict()
    del_b=dict()
    del_h=dict()

    '''for output layer set values based on cross entropy or mse loss'''
    if(self.loss=="cross_entropy"):
      del_a[self.num_of_layers-1]=-(y-y_cap)
    elif(self.loss=="mean_squared_error"):
      del_a[self.num_of_layers-1]=(y_cap-y)*y_cap*(1-y_cap)
    
    '''iterate till input layer'''
    for k in range(self.num_of_layers-1,0,-1):
      del_w[k]=np.matmul(del_a[k],h[k-1].T)
      del_b[k]=np.sum(del_a[k],axis=1)
      del_h[k-1]=np.matmul(self.W[k].T,del_a[k])
      if k>1:
        if(self.activation=="sigmoid"):
          del_a[k-1]=np.multiply(del_h[k-1],Activation.grad_sigmoid(a[k-1]))
        elif(self.activation=="tanh"):
          del_a[k-1]=np.multiply(del_h[k-1],Activation.grad_tanh(a[k-1]))
        elif(self.activation=="identity"):
          del_a[k-1]=np.multiply(del_h[k-1],Activation.grad_identity(a[k-1]))
        else:
          del_a[k-1]=np.multiply(del_h[k-1],Activation.grad_relu(a[k-1]))

    return (del_w,del_b)

  def stochastic_gradient_descent(self,batch_size):
    '''
      Parameters:
        batch_size
      Returns :
        W : weight
        B : Bias
      Function:
        performs stochastic gradient descent
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    for iter in range(self.epochs):
      predictions=list()
      for i in range(0,self.x_train.shape[0],batch_size):
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(self.W,self.B,x)
        ret=self.backward_propagation(H,A,y,y_cap)
        self.W,self.B=UpdateParameters.update_parameters(self.W,self.B,self.eta,ret[0],ret[1],self.wd)

      predictions=list()
      for i in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[i])
        predictions.append(val)
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for i in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[i])
        predictions.append(val)
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])
      
    return self.W,self.B

  def momentum_gradient_descent(self,beta,batch_size):
    '''
      Parameters:
        batch_size
        beta : momentum value taken in each epoch
      Returns :
        W : weight
        B : Bias
      Function:
        performs momentum based gradient descent
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    prev_uw,prev_ub=self.initGrads()
    for iter in range(self.epochs):
      predictions=list()
      for i in range(0,self.x_train.shape[0],batch_size):
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(self.W,self.B,x)
        ret=self.backward_propagation(H,A,y,y_cap)
        uw,ub=dict(),dict()
        for i in range(1,len(ret[0])):
          uw[i]=beta*prev_uw[i]+self.eta*ret[0][i]
          ub[i]=beta*prev_ub[i]+self.eta*ret[1][i]

        self.W,self.B=UpdateParameters.update_parameters_mgd(self.W,self.B,uw,ub,self.eta,self.wd)
        prev_uw=uw
        prev_ub=ub

      predictions=list()
      for i in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[i])
        predictions.append(val)
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for i in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[i])
        predictions.append(val)
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])

    return self.W,self.B

  def nestrov_gradient_descent(self,beta,batch_size):
    '''
      Parameters:
        batch_size
        beta : momentum value taken in each epoch
      Returns :
        W : weight
        B : Bias
      Function:
        performs nestrov accelerated gradient descent
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    prev_uw,prev_ub=self.initGrads()

    for iter in range(self.epochs):
      predictions=list()
      uw,ub=dict(),dict()
      for i in range(1,len(prev_uw)):
        uw[i]=beta*prev_uw[i]
        ub[i]=beta*prev_ub[i]

      w,b=UpdateParameters.update_parameters_mgd(self.W,self.B,uw,ub,self.eta,self.wd)

      for i in range(0,self.x_train.shape[0],batch_size):
        original_selfW,original_selfB=dict(),dict()
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(w,b,x)
        for j in range(1,len(self.W)+1):
          original_selfW[j]=self.W[j]
          self.W[j]=w[j]
          original_selfB[j]=self.B[j]
          self.B[j]=b[j]
        ret=self.backward_propagation(H,A,y,y_cap)
        for j in range(1,len(self.W)+1):
          self.W[j]=original_selfW[j]
          self.B[j]=original_selfB[j]

        for j in range(1,len(ret[0])+1):
          uw[j]=beta*prev_uw[j]+self.eta*ret[0][j]
          ub[j]=beta*prev_ub[j]+self.eta*ret[1][j]

        self.W,self.B=UpdateParameters.update_parameters_mgd(self.W,self.B,uw,ub,self.eta,self.wd)
        prev_uw=uw
        prev_ub=ub

      predictions=list()
      for i in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[i])
        predictions.append(val)
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for i in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[i])
        predictions.append(val)
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])

    return self.W,self.B


  def rmsprop(self,beta,eps,batch_size):
    '''
      Parameters:
        batch_size
        beta : learning rate updation parameter
        eps : epsilon (correction value)
      Returns :
        W : weight
        B : Bias
      Function:
        performs rmsprop
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    vw,vb=self.initGrads()

    for iter in range(self.epochs):
      predictions=list()
      for i in range(0,self.x_train.shape[0],batch_size):
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(self.W,self.B,x)
        _ret=self.backward_propagation(H,A,y,y_cap)

        ret=list()
        ret.append(_ret[0])
        ret.append(_ret[1])
        for j in range(1,len(ret[0])+1):
          vw[j]=(beta*vw[j])+((1-beta)*(np.square(ret[0][j])))
          vb[j]=(beta*vb[j])+((1-beta)*(np.square(ret[1][j])))

        self.W,self.B=UpdateParameters.update_parameters_rms(self.W,self.B,self.eta,vw,vb,ret[0],ret[1],eps,self.wd)

      predictions=list()
      for i in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[i])
        predictions.append(val)
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for i in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[i])
        predictions.append(val)
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])

    return self.W,self.B


  def adam(self,beta1,beta2,eps,batch_size):
    '''
      Parameters:
        batch_size
        beta 1, beta 2 : parameters used for updation
        eps : epsilon (correction value)
      Returns :
        W : weight
        B : Bias
      Function:
        performs adam
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    vw,vb=self.initGrads()
    mw,mb=self.initGrads()
    mw_hat,mb_hat=self.initGrads()
    vw_hat,vb_hat=self.initGrads()

    for iter in range(self.epochs):
      predictions=list()
      for i in range(0,self.x_train.shape[0],batch_size):
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(self.W,self.B,x)
        _ret=self.backward_propagation(H,A,y,y_cap)

        ret=list()
        ret.append(_ret[0])
        ret.append(_ret[1])
        for j in range(1,len(ret[0])+1):
          mw[j]=(beta1*mw[j])+((1-beta1)*ret[0][j])
          mb[j]=(beta1*mb[j])+((1-beta1)*ret[1][j])
          vw[j]=(beta2*vw[j])+((1-beta2)*(np.square(ret[0][j])))
          vb[j]=(beta2*vb[j])+((1-beta2)*(np.square(ret[1][j])))

        for j in range(1,len(ret[0])+1):
          mw_hat[j]=mw[j]/(1-np.power(beta1,iter+1))
          mb_hat[j]=mb[j]/(1-np.power(beta1,iter+1))
          vw_hat[j]=vw[j]/(1-np.power(beta2,iter+1))
          vb_hat[j]=vb[j]/(1-np.power(beta2,iter+1))

        self.W,self.B=UpdateParameters.update_parameters_adam(self.W,self.B,self.eta,mw_hat,mb_hat,vw_hat,vb_hat,eps,self.wd)

      predictions=list()
      for j in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[j])
        predictions.append(val)
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for j in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[j])
        predictions.append(val)
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])

    return self.W,self.B


  def nadam(self,beta1,beta2,eps,batch_size):
    '''
      Parameters:
        batch_size
        beta 1, beta 2 : parameters used for updation
        eps : epsilon (correction value)
      Returns :
        W : weight
        B : Bias
      Function:
        performs nadam
    '''
    trainLossPerEpoch=list()
    trainAccuracyPerEpoch=list()
    valLossPerEpoch=list()
    valAccuracyPerEpoch=list()

    vw,vb=self.initGrads()
    mw,mb=self.initGrads()
    mw_hat,mb_hat=self.initGrads()
    vw_hat,vb_hat=self.initGrads()

    for iter in range(self.epochs):
      predictions=list()
      for i in range(0,self.x_train.shape[0],batch_size):
        x=self.x_train[i:i+batch_size].T
        y=self.y_train[i:i+batch_size].T
        A,H,y_cap=self.forward_propagation(self.W,self.B,x)
        _ret=self.backward_propagation(H,A,y,y_cap)

        ret=list()
        ret.append(_ret[0])
        ret.append(_ret[1])
        for i in range(1,len(ret[0])+1):
          mw[i]=(beta1*mw[i])+((1-beta1)*ret[0][i])
          mb[i]=(beta1*mb[i])+((1-beta1)*ret[1][i])
          vw[i]=(beta2*vw[i])+((1-beta2)*(np.square(ret[0][i])))
          vb[i]=(beta2*vb[i])+((1-beta2)*(np.square(ret[1][i])))

        for i in range(1,len(ret[0])+1):
          mw_hat[i]=mw[i]/(1-np.power(beta1,i+1))
          mb_hat[i]=mb[i]/(1-np.power(beta1,i+1))
          vw_hat[i]=vw[i]/(1-np.power(beta2,i+1))
          vb_hat[i]=vb[i]/(1-np.power(beta2,i+1))

        self.W,self.B=UpdateParameters.update_parameters_nadam(self.W,self.B,self.eta,mw_hat,mb_hat,vw_hat,vb_hat,beta1,beta2,ret[0],ret[1],eps,self.wd)
        
      predictions=list()
      for i in range(self.x_train.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_train[i])
        predictions.append(np.array(val))
      if(self.loss=="cross_entropy"):
        trainLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_train,np.array(predictions))/self.x_train.shape[0])
      else:
        trainLossPerEpoch.append(Loss.trainMseLoss(self.y_train,np.array(predictions)))
      trainAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_train,np.array(predictions)))
      predictions=list()
      for i in range(self.x_val.shape[0]):
        _,_,val=self.forward_propagation_test(self.W,self.B,self.x_val[i])
        predictions.append(np.array(val))
      valAccuracyPerEpoch.append(Accuracy.trainAccuracy(self.y_val,np.array(predictions)))
      if(self.loss=="cross_entropy"):
        valLossPerEpoch.append(Loss.trainCrossEntropyLoss(self.y_val,np.array(predictions))/self.x_val.shape[0])
      else:
        valLossPerEpoch.append(Loss.trainMseLoss(self.y_val,np.array(predictions)))
      print("\n===================================================================================================================")
      print("Epoch = {}".format(iter+1))
      print("Training Accuracy = {}".format(trainAccuracyPerEpoch[-1]))
      print("Validation Accuracy = {}".format(valAccuracyPerEpoch[-1]))
      print("Training Loss = {}".format(trainLossPerEpoch[-1]))
      print("Validation Loss = {}".format(valLossPerEpoch[-1]))
      if(self.mnistRun==False):
        wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(iter+1)})

    if(self.mnistRun==True):
      print("\n\nTraining Accuracy : ",trainAccuracyPerEpoch[-1])
      print("Validation Accuracy : ",valAccuracyPerEpoch[-1])
    return self.W,self.B


  def modelFitting(self,mom,beta,beta1,beta2,eps,optimizer,batch_size,test,confusion_matrix,dataset_name):
    '''
      Parameters:
        batch_size
        beta, beta 1, beta 2 : parameters used for updation
        eps : epsilon (correction value)
        optimizer : name of the particular optimizer
      Returns :
        None
      Function:
        calls particular optimizer function based on the algorithm called
    '''

    '''generates run name and logs it to wandb'''
    run="LR_{}_OP_{}_EP_{}_BS_{}_INIT_{}_HL_{}_NHL_{}_AC_{}_WD_{}_LS_{}".format(self.eta,optimizer,self.epochs,batch_size,self.initialization,self.hls,self.neurons_in_hl,self.activation,self.wd,self.loss)
    print("run name = {}".format(run))
    wandb.run.name=run

    if(optimizer=="sgd"):
      w,b=self.stochastic_gradient_descent(batch_size)
    elif(optimizer=="momentum"):
      w,b=self.momentum_gradient_descent(mom,batch_size)
    elif(optimizer=="nag"):
      w,b=self.nestrov_gradient_descent(mom,batch_size)
    elif(optimizer=="rmsprop"):
      w,b=self.rmsprop(beta,eps,batch_size)
    elif(optimizer=="adam"):
      w,b=self.adam(beta1,beta2,eps,batch_size)
    elif(optimizer=="nadam"):
      w,b=self.nadam(beta1,beta2,eps,batch_size)
    
    '''do forward propagation on the test data using the weights obtained above and report accuracy'''
    yPred=list()
    for i in range(len(self.y_test)):
      _,_,y_cap=self.forward_propagation_test(w,b,self.x_test[i])
      yPred.append(np.array(y_cap))
    
    if(test==1):
      print("\n\nTest Accuracy : ",Accuracy.testAccuracy(self.y_test,yPred))

    if(confusion_matrix==1):
      output_class=list()
      if(dataset_name=="fashion_mnist"):
        output_class=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
      else:
        output_class=["0","1","2","3","4","5","6","7","8","9"]
      true_class_labels=list()
      predicted_class_labels=list()
      for i in range(self.y_test.shape[0]):
        true_class_labels.append(output_class[np.argmax(self.y_test[i])])
        predicted_class_labels.append(output_class[np.argmax(yPred[i])])
      wandb.sklearn.plot_confusion_matrix(true_class_labels,predicted_class_labels,output_class)


  def modelFittingForMnist(self,mom,beta,beta1,beta2,eps,optimizer,batch_size):
    '''
      Parameters:
        batch_size
        beta, beta 1, beta 2 : parameters used for updation
        eps : epsilon (correction value)
        optimizer : name of the particular optimizer
      Returns :
        None
      Function:
        calls particular optimizer function based on the algorithm called. this function is called to report the accuracies of the three best config for mnist dataset (question 10)
    '''
    w,b=dict(),dict()

    self.mnistRun=True
    if(optimizer=="sgd"):
      w,b=self.stochastic_gradient_descent(batch_size)
    elif(optimizer=="momentum"):
      w,b=self.momentum_gradient_descent(mom,batch_size)
    elif(optimizer=="nag"):
      w,b=self.nestrov_gradient_descent(mom,batch_size)
    elif(optimizer=="rmsprop"):
      w,b=self.rmsprop(beta,eps,batch_size)
    elif(optimizer=="adam"):
      w,b=self.adam(beta1,beta2,eps,batch_size)
    elif(optimizer=="nadam"):
      w,b=self.nadam(beta1,beta2,eps,batch_size)
    self.mnistRun=False

    return w,b