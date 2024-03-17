import numpy as np

class ParamaterInitialization:

  def initializeW(initialization,num_of_layers,neurons_in_hl,x_train_shape,y_train_shape):
    '''
      Parameters:
        initialization : type of initialization (random or xavier)
        num_of_layers : number of hidden layers (required for getting the size of the weight matrix)
        neurons_in_hl : number of neurons per hidden layer (required for getting the size of weight matrix of each layer)
        x_train_shape : length of the training input data vector
        y_train_shape : length of the training output data vector
      Returns :
        array having matrices with values 0
      Function:
        Creates weight matrix for each hidden layer and initialize them to 0
    '''
    weights=dict()
    if(initialization=="random"):
      '''input layer'''
      weights[1]=np.random.randn(neurons_in_hl,x_train_shape).astype(np.float64)
      '''hidden layers'''
      for i in range(2,num_of_layers-1):
        weights[i]=np.random.randn(neurons_in_hl,neurons_in_hl).astype(np.float64)
      '''output layer'''
      weights[num_of_layers-1]=np.random.randn(y_train_shape,neurons_in_hl).astype(np.float64)

    else:
      '''input layer'''
      weights[1]=np.random.randn(neurons_in_hl,x_train_shape)*np.sqrt(2/(x_train_shape+neurons_in_hl)).astype(np.float64)
      '''hidden layers'''
      for i in range(2,num_of_layers-1):
        weights[i]=np.random.randn(neurons_in_hl, neurons_in_hl)*np.sqrt(2/(neurons_in_hl+neurons_in_hl)).astype(np.float64)
      '''output layer'''
      weights[num_of_layers-1]=np.random.randn(y_train_shape,neurons_in_hl)*np.sqrt(2/(y_train_shape+neurons_in_hl)).astype(np.float64)

    return weights

  def initializeB(initialization,num_of_layers,neurons_in_hl,x_train_shape,y_train_shape):
    '''
      Parameters:
        initialization : type of initialization (random or xavier)
        num_of_layers : number of hidden layers (required for getting the size of the bias matrix)
        neurons_in_hl : number of neurons per hidden layer (required for getting the size of bias matrix of each layer)
        x_train_shape : length of the training input data vector
        y_train_shape : length of the training output data vector
      Returns :
        array having matrices with values 0
      Function:
        Creates bias matrix for each hidden layer and initialize them to 0
    '''
    biases=dict()
    if(initialization=="random"):
      '''input and hidden layers'''
      for i in range(1,num_of_layers-1):
        biases[i]=np.random.randn(neurons_in_hl).astype(np.float64)
      '''output layer'''
      biases[num_of_layers-1]=np.random.randn(y_train_shape).astype(np.float64)

    else:
      '''input and hidden layers'''
      for i in range(1,num_of_layers-1):
        biases[i]=np.random.randn(neurons_in_hl).astype(np.float64)
      '''output layer'''
      biases[num_of_layers-1]=np.random.randn(y_train_shape).astype(np.float64)

    return biases