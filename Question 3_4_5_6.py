import wandb
from NeuralNetwork import FeedForwardNeuralNetwork,x_train,y_train,x_val,y_val

'''login to wandb to generate plot'''
wandb.login()

def main():
    '''initialize to project adn create a config'''
    wandb.init(project="Pritam CS6910 - Assignment 1")
    config=wandb.config
   
    '''create an object of the FeedForwardNeuralNetwork class which has all the required functions
      pass the parameters to the constructor as a sweep value. this will change the values with each run of the sweep.
      call the fitting fucntion and pass the optimizer as a parameter
    '''
    Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,hls=config.number_of_hidden_layers,neurons_in_hl=config.neurons_in_each_hidden_layers,activation=config.activation_function,initialization=config.initialization_technique,epochs=config.number_of_epochs,eta=config.learning_rate,wd=config.weight_decay,loss=config.loss_type)
    Model.modelFitting(beta=config.beta_value,beta1=0.9,beta2=0.999,eps=1e-5,optimizer=config.optimizer_technique,batch_size=config.batch_size)


'''sweep configuration'''
configuration_values = {
    'method': 'bayes',
    'name': 'ACCURACY AND LOSS',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
        },
    'parameters': {
        'initialization_technique': {'values': ['xavier','random']},
        'number_of_hidden_layers' : {'values' : [3,4,5]},
        'neurons_in_each_hidden_layers' : {'values' : [32,64,128]},

        'learning_rate': {'values':[1e-1,5e-1,1e-2,1e-3,5e-3,1e-4]},
        'beta_value' : {'values' : [0.9,0.999]},
        'optimizer_technique' : {'values' : ['sgd','momentum','rmsprop','adam','nadam','nag']},

        'batch_size': {'values': [16,32,64,128]},
        'number_of_epochs': {'values': [5,10,20,30]},
        'loss_type' : {'values' : ['cross_entropy']},
        'activation_function' : {'values' : ['identity','sigmoid','ReLU','tanh']},
        'weight_decay' : {'values' : [0, 0.0005,0.5]}
       }
    }

'''create a sweep id in the current project'''
sweep_id=wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 1')
'''generate a sweep agent to run the sweep'''
wandb.agent(sweep_id,function=main,count=150)
wandb.finish()