from keras.datasets import mnist,fashion_mnist
import wandb
import numpy as np
from NeuralNetworkForTrainPyImplementation import FeedForwardNeuralNetwork
import warnings
warnings.filterwarnings("ignore")
import argparse

'''login to wandb to generate plot'''
wandb.login()

def arguments():
    '''
      Parameters:
        None
      Returns :
        A parser object
      Function:
        Does command line argument parsing and returns the arguments passed
    '''
    commandLineArgument=argparse.ArgumentParser(description='Model Parameters')
    commandLineArgument.add_argument('-wp','--wandb_project',help="Project name used to track experiments in Weights & Biases dashboard")
    commandLineArgument.add_argument('-we','--wandb_entity',help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    commandLineArgument.add_argument('-d','--dataset',help="choices: ['mnist', 'fashion_mnist']")
    commandLineArgument.add_argument('-e','--epochs',type = int,help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch_size',type = int,help="Batch size used to train neural network")
    commandLineArgument.add_argument('-l','--loss',help="choices: ['mean_squared_error', 'cross_entropy']")
    commandLineArgument.add_argument('-o','--optimizer',help="choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
    commandLineArgument.add_argument('-lr','--learning_rate',type = float,help="Learning rate used to optimize model parameters")
    commandLineArgument.add_argument('-m','--momentum',type = float,help="Momentum used by momentum and nag optimizers")
    commandLineArgument.add_argument('-beta','--beta',type = float,help="Beta used by rmsprop optimizer")
    commandLineArgument.add_argument('-beta1','--beta1',type = float,help="Beta1 used by adam and nadam optimizers")
    commandLineArgument.add_argument('-beta2','--beta2',type = float,help="Beta2 used by adam and nadam optimizers")
    commandLineArgument.add_argument('-eps','--epsilon',type = float,help="Epsilon used by optimizers")
    commandLineArgument.add_argument('-w_d','--weight_decay',type = float,help="Weight decay used by optimizers")
    commandLineArgument.add_argument('-w_i','--weight_init',help="choices: ['random', 'Xavier']")
    commandLineArgument.add_argument('-nhl','--num_layers',type = int,help="Number of hidden layers used in feedforward neural network")
    commandLineArgument.add_argument('-sz','--hidden_size',type = int,help="Number of hidden neurons in a feedforward layer")
    commandLineArgument.add_argument('-a','--activation',help="choices: ['identity', 'sigmoid', 'tanh', 'ReLU']")

    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    project_name='Pritam CS6910 - Assignment 1'
    entity_name='cs23m051'
    learning_rate=1e-3
    mom=0.9
    beta=0.9
    beta1=0.9
    beta2=0.999
    epsilon=1e-5
    optimizer="nadam"
    batch_size=32
    loss="cross_entropy"
    epochs=30
    wd=.0
    dataset_name="fashion_mnist"
    wi="xavier"
    hls=4
    nhls=64
    act="relu"

    '''call to argument function to get the arguments'''
    argumentsPassed=arguments()

    '''checking if a particular argument is passed thorugh commadn line or not and updating the values accordingly'''
    if argumentsPassed.wandb_project is not None:
        project_name=argumentsPassed.wandb_project
    if argumentsPassed.wandb_entity is not None:
        entity_name=argumentsPassed.wandb_entity
    if argumentsPassed.dataset is not None:
        dataset_name=argumentsPassed.dataset
    if argumentsPassed.learning_rate is not None:
        learning_rate=argumentsPassed.learning_rate
    if argumentsPassed.momentum is not None:
        mom=argumentsPassed.momentum
    if argumentsPassed.beta is not None:
        beta=argumentsPassed.beta
    if argumentsPassed.beta1 is not None:
        beta1=argumentsPassed.beta1
    if argumentsPassed.beta2 is not None:
        beta2=argumentsPassed.beta2
    if argumentsPassed.epsilon is not None:
        epsilon=argumentsPassed.epsilon
    if argumentsPassed.optimizer is not None:
        optimizer=argumentsPassed.optimizer
    if argumentsPassed.batch_size is not None:
        batch_size=argumentsPassed.batch_size
    if argumentsPassed.loss is not None:
        loss=argumentsPassed.loss
    if argumentsPassed.epochs is not None:
        epochs=argumentsPassed.epochs
    if argumentsPassed.weight_decay is not None:
        wd=argumentsPassed.weight_decay
    if argumentsPassed.weight_init is not None:
        wi=argumentsPassed.weight_init
    if argumentsPassed.num_layers is not None:
        hls=argumentsPassed.num_layers
    if argumentsPassed.hidden_size is not None:
        nhls=argumentsPassed.hidden_size
    if argumentsPassed.activation is not None:
        act=argumentsPassed.activation
    
    '''setting the dataset and reading the corresponding train and test data'''
    if(dataset_name=="fashion_mnist"):
        (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    else:
        (x_train,y_train),(x_test,y_test)=mnist.load_data()
    
    '''splitting into validation and train data'''
    validation_ratio=0.1
    num_validation_samples=int(validation_ratio*x_train.shape[0])
    validation_indices=np.random.choice(x_train.shape[0],num_validation_samples,replace=False)

    x_val,y_val=x_train[validation_indices],y_train[validation_indices]
    x_train, y_train=np.delete(x_train,validation_indices,axis=0),np.delete(y_train,validation_indices,axis=0)

    '''initializing to the project'''
    wandb.init(project=project_name,entity=entity_name)

    '''calling the functions with the parameters'''
    Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=hls,neurons_in_hl=nhls,activation=act,initialization=wi,epochs=epochs,eta=learning_rate,wd=wd,loss=loss)
    Model.modelFitting(mom=mom,beta=beta,beta1=beta1,beta2=beta2,eps=epsilon,optimizer=optimizer,batch_size=batch_size)

if __name__ == '__main__':
    main()