import wandb
from NeuralNetwork import FeedForwardNeuralNetwork,x_train,y_train,x_val,y_val,y_test,x_test
import numpy as np
from AccuracyCalculation import Accuracy

Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,hls=4,neurons_in_hl=128,activation="relu",initialization="xavier",epochs=30,eta=1e-2,wd=0,loss="cross_entropy")
w,b=Model.modelFittingForBestConfig(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="nadam",batch_size=32)

y_test1=Model.one_hot_encoding(y_test)
x_test1=Model.input_flattening(x_test)

yPred=list()
for i in range(len(y_test1)):
  _,_,y_cap=Model.forward_propagation_test(w,b,x_test1[i])
  yPred.append(np.array(y_cap))

test_accuracy=Accuracy.testAccuracy(y_test1,yPred)
print("Test accuracy for my best model = {}".format(test_accuracy))

output_class=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

true_class_labels=list()
predicted_class_labels=list()

for i in range(y_test1.shape[0]):
  true_class_labels.append(output_class[np.argmax(y_test1[i])])
  predicted_class_labels.append(output_class[np.argmax(yPred[i])])

wandb.init(project='Pritam CS6910 - Assignment 1')
wandb.sklearn.plot_confusion_matrix(true_class_labels,predicted_class_labels,output_class)
wandb.finish()