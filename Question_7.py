import wandb
from NeuralNetwork import FeedForwardNeuralNetwork,x_train,y_train,x_val,y_val,y_test,x_test
import numpy as np
from AccuracyCalculation import Accuracy

'''login to wandb to generate plot'''
wandb.login()

'''
  create an object of the FeedForwardNeuralNetwork class which has all the required functions
  pass the parameters to the constructor and set them to the values corresponding to the best model configuration
  call the fitting fucntion
'''
Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,hls=4,neurons_in_hl=64,activation="ReLU",initialization="xavier",epochs=30,eta=1e-3,wd=0,loss="cross_entropy")
weights,biases=Model.modelFittingForBestConfig(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="nadam",batch_size=32)

'''one hot encode the test output and flatten the test input'''
y_test1=Model.one_hot_encoding(y_test)
x_test1=Model.input_flattening(x_test)

'''perform forward propagation on the test input to find the test predictions by using the weights and biases obtained from the backpropagation function'''
yPred=list()
for i in range(len(y_test1)):
  _,_,y_cap=Model.forward_propagation_test(weights,biases,x_test1[i])
  yPred.append(np.array(y_cap))

'''call accuracy function on the predictions'''
test_accuracy=Accuracy.testAccuracy(y_test1,yPred)
print("Test accuracy for my best model = {}".format(test_accuracy))


'''code for generating confusion matrix'''
output_class=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
true_class_labels=list()
predicted_class_labels=list()

'''get class labels for the true outputs and their corresponding predictions'''
for i in range(y_test1.shape[0]):
  true_class_labels.append(output_class[np.argmax(y_test1[i])])
  predicted_class_labels.append(output_class[np.argmax(yPred[i])])

'''plot the confusion matrix using sklearn'''
wandb.init(project='Pritam CS6910 - Assignment 1')
wandb.sklearn.plot_confusion_matrix(true_class_labels,predicted_class_labels,output_class)
wandb.finish()