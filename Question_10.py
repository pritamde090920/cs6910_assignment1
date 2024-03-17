import wandb
import numpy as np
from keras.datasets import mnist
from NeuralNetworkForTrainPyImplementation import FeedForwardNeuralNetwork
from AccuracyCalculation import Accuracy


'''load the mnist dataset'''
(x_train,y_train),(x_test,y_test)=mnist.load_data()

'''split the dataset into 10% validation and 90% training'''
validation_ratio=0.1
num_validation_samples=int(validation_ratio*x_train.shape[0])
validation_indices=np.random.choice(x_train.shape[0],num_validation_samples,replace=False)

x_val,y_val=x_train[validation_indices],y_train[validation_indices]
x_train, y_train=np.delete(x_train,validation_indices,axis=0),np.delete(y_train,validation_indices,axis=0)


'''1st configuration'''
'''get the weight and bias by running the fitting function for the configuration'''
Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=4,neurons_in_hl=64,activation="relu",initialization="xavier",epochs=30,eta=1e-3,wd=0,loss="cross_entropy")
weights,biases=Model.modelFittingForMnist(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="nadam",batch_size=32)

x_test=Model.input_flattening(x_test)
y_test=Model.one_hot_encoding(y_test)

'''run forward propagation on the test datat to generate predictions using the weights and biases and calculate the accuracy'''
yPred=list()
for i in range(len(y_test)):
  _,_,y_cap=Model.forward_propagation_test(weights,biases,x_test[i])
  yPred.append(np.array(y_cap))

print("Test Accuracy for 1st config",Accuracy.testAccuracy(y_test,yPred))


'''2nd configuration'''
'''get the weight and bias by running the fitting function for the configuration'''
Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=4,neurons_in_hl=64,activation="tanh",initialization="xavier",epochs=30,eta=1e-4,wd=0,loss="cross_entropy")
weights,biases=Model.modelFittingForMnist(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="nadam",batch_size=32)

x_test=Model.input_flattening(x_test)
y_test=Model.one_hot_encoding(y_test)

'''run forward propagation on the test datat to generate predictions using the weights and biases and calculate the accuracy'''
yPred=list()
for i in range(len(y_test)):
  _,_,y_cap=Model.forward_propagation_test(weights,biases,x_test[i])
  yPred.append(np.array(y_cap))

print("Test Accuracy for 2nd config",Accuracy.testAccuracy(y_test,yPred))



'''3rd configuration'''
'''get the weight and bias by running the fitting function for the configuration'''
Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=4,neurons_in_hl=64,activation="tanh",initialization="xavier",epochs=30,eta=1e-3,wd=0,loss="cross_entropy")
weights,biases=Model.modelFittingForMnist(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="adam",batch_size=32)

x_test=Model.input_flattening(x_test)
y_test=Model.one_hot_encoding(y_test)

'''run forward propagation on the test datat to generate predictions using the weights and biases and calculate the accuracy'''
yPred=list()
for i in range(len(y_test)):
  _,_,y_cap=Model.forward_propagation_test(weights,biases,x_test[i])
  yPred.append(np.array(y_cap))

print("Test Accuracy for 3rd config",Accuracy.testAccuracy(y_test,yPred))