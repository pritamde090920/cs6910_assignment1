import numpy as np

class ParamaterInitialization:
  def initializeW(initialization,num_of_layers,neurons_in_hl,x_train_shape,y_train_shape):
    w=dict()
    if(initialization=="random"):
      w[1]=np.random.randn(neurons_in_hl,x_train_shape).astype(np.float64)
      for i in range(2,num_of_layers-1):
        w[i]=np.random.randn(neurons_in_hl,neurons_in_hl).astype(np.float64)
      w[num_of_layers-1]=np.random.randn(y_train_shape,neurons_in_hl).astype(np.float64)

    else:
      w[1]=np.random.randn(neurons_in_hl,x_train_shape)*np.sqrt(2/(x_train_shape+neurons_in_hl)).astype(np.float64)
      for i in range(2,num_of_layers-1):
        w[i]=np.random.randn(neurons_in_hl, neurons_in_hl)*np.sqrt(2/(neurons_in_hl+neurons_in_hl)).astype(np.float64)
      w[num_of_layers-1]=np.random.randn(y_train_shape,neurons_in_hl)*np.sqrt(2/(y_train_shape+neurons_in_hl)).astype(np.float64)

    return w

  def initializeB(initialization,num_of_layers,neurons_in_hl,x_train_shape,y_train_shape):
    b=dict()
    if(initialization=="random"):
      for i in range(1,num_of_layers-1):
        b[i]=np.random.randn(neurons_in_hl).astype(np.float64)
      b[num_of_layers-1]=np.random.randn(y_train_shape).astype(np.float64)

    else:
      for i in range(1,num_of_layers-1):
        b[i]=np.random.randn(neurons_in_hl).astype(np.float64)
      b[num_of_layers-1]=np.random.randn(y_train_shape).astype(np.float64)

    return b