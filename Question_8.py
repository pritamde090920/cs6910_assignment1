import wandb
import numpy as np
from NeuralNetwork import FeedForwardNeuralNetwork,x_train,y_train,x_val,y_val

'''login to wandb to generate plot'''
wandb.login()

def main():
    wandb.init(project="Pritam CS6910 - Assignment 1")
    config=wandb.config
    '''
        create an object of the FeedForwardNeuralNetwork class which has all the required functions
        pass the parameters to the constructor as a sweep value. this will change the values with each run of the sweep.
        call the fitting fucntion
    '''
    Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,hls=config.number_of_hidden_layers,neurons_in_hl=config.neurons_in_each_hidden_layers,activation=config.activation_function,initialization=config.initialization_technique,epochs=config.number_of_epochs,eta=config.learning_rate,wd=config.weight_decay,loss=config.loss_type)
    Model.modelFittingForCeVsMse(beta=config.beta_value,beta1=0.9,beta2=0.999,eps=1e-5,optimizer=config.optimizer_technique,batch_size=config.batch_size)


'''sweep configuration. all the parameteres have single value. only loss has cross entropy and mse, because we want to generate plots on both'''
configuration_values = {
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
        'activation_function' : {'values' : ['ReLU']},
        'weight_decay' : {'values' : [0]}
       }
    }

'''create sweep id and call an agent to run the sweep'''
sweep_id = wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 1')
wandb.agent(sweep_id,function=main,count=2)
wandb.finish()