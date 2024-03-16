import wandb
import numpy as np
from keras.datasets import mnist
from NeuralNetworkForTrainPyImplementation import FeedForwardNeuralNetwork
from AccuracyCalculation import Accuracy


(x_train,y_train),(x_test,y_test)=mnist.load_data()
validation_ratio=0.1
num_validation_samples=int(validation_ratio*x_train.shape[0])
validation_indices=np.random.choice(x_train.shape[0],num_validation_samples,replace=False)

x_val,y_val=x_train[validation_indices],y_train[validation_indices]
x_train, y_train=np.delete(x_train,validation_indices,axis=0),np.delete(y_train,validation_indices,axis=0)




# def main():
#   wandb.init(project="Pritam CS6910 - Assignment 1")
#   config=wandb.config

#   Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=config.number_of_hidden_layers,neurons_in_hl=config.neurons_in_each_hidden_layers,activation=config.activation_function,initialization=config.initialization_technique,epochs=config.number_of_epochs,eta=config.learning_rate,wd=config.weight_decay,loss=config.loss_type)
#   Model.modelFittingForCeVsMse(beta=config.beta_value,beta1=0.9,beta2=0.999,eps=1e-5,optimizer=config.optimizer_technique,batch_size=config.batch_size)

Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=4,neurons_in_hl=64,activation="relu",initialization="xavier",epochs=30,eta=1e-3,wd=0,loss="cross_entropy")
w,b=Model.modelFittingForMnist(beta=0.9,beta1=0.9,beta2=0.999,eps=1e-5,optimizer="nadam",batch_size=32)

x_test=Model.input_flattening(x_test)
y_test=Model.one_hot_encoding(y_test)

yPred=list()
for i in range(len(y_test)):
  _,_,y_cap=Model.forward_propagation_test(w,b,x_test[i])
  yPred.append(np.array(y_cap))

print("Test Accuracy",Accuracy.testAccuracy(y_test,yPred))

#1st config
# sweep_configuration = {
#     'method': 'grid',
#     'name': 'CROSS ENTROPY VS MSE',
#     'metric': {
#         'goal': 'maximize',
#         'name': 'validation_accuracy'
#         },
#     'parameters': {
#         'initialization_technique': {'values': ['xavier']},
#         'number_of_hidden_layers' : {'values' : [4]},
#         'neurons_in_each_hidden_layers' : {'values' : [64]},

#         'learning_rate': {'values':[1e-3]},
#         'beta_value' : {'values' : [0.9]},
#         'optimizer_technique' : {'values' : ['nadam']},

#         'batch_size': {'values': [32]},
#         'number_of_epochs': {'values': [30]},
#         'loss_type' : {'values' : ['cross_entropy']},
#         'activation_function' : {'values' : ['relu']},
#         'weight_decay' : {'values' : [0]}
#        }
#     }

# sweep_id = wandb.sweep(sweep=sweep_configuration,project='Pritam CS6910 - Assignment 1')
# wandb.agent(sweep_id,function=main,count=1)
# wandb.finish()