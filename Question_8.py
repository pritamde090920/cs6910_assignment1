import wandb
import numpy as np
from NeuralNetwork import FeedForwardNeuralNetwork,x_train,y_train,x_val,y_val


def main():
  wandb.init(project="Pritam CS6910 - Assignment 1")
  config=wandb.config

  Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,hls=config.number_of_hidden_layers,neurons_in_hl=config.neurons_in_each_hidden_layers,activation=config.activation_function,initialization=config.initialization_technique,epochs=config.number_of_epochs,eta=config.learning_rate,wd=config.weight_decay,loss=config.loss_type)
  Model.modelFittingForCeVsMse(beta=config.beta_value,beta1=0.9,beta2=0.999,eps=1e-5,optimizer=config.optimizer_technique,batch_size=config.batch_size)


# #1st config
sweep_configuration = {
    'method': 'grid',
    'name': 'CROSS ENTROPY VS MSE',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
        },
    'parameters': {
        'initialization_technique': {'values': ['xavier']},
        'number_of_hidden_layers' : {'values' : [4]},
        'neurons_in_each_hidden_layers' : {'values' : [64]},

        'learning_rate': {'values':[1e-3]},
        'beta_value' : {'values' : [0.9]},
        'optimizer_technique' : {'values' : ['nadam']},

        'batch_size': {'values': [32]},
        'number_of_epochs': {'values': [30]},
        'loss_type' : {'values' : ['cross_entropy','mse']},
        'activation_function' : {'values' : ['relu']},
        'weight_decay' : {'values' : [0]}
       }
    }

sweep_id = wandb.sweep(sweep=sweep_configuration,project='Pritam CS6910 - Assignment 1')
wandb.agent(sweep_id,function=main,count=2)
wandb.finish()