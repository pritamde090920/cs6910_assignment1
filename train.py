from keras.datasets import mnist,fashion_mnist
import wandb
import numpy as np
from NeuralNetworkForTrainPyImplementation import FeedForwardNeuralNetwork
import warnings
warnings.filterwarnings("ignore")
import argparse


parser=argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('-wp','--wandb_project')
parser.add_argument('-we','--wandb_entity')
parser.add_argument('-d','--dataset')
parser.add_argument('-e','--epochs',type = int)
parser.add_argument('-b','--batch_size',type = int)
parser.add_argument('-l','--loss')
parser.add_argument('-o','--optimizer')
parser.add_argument('-lr','--learning_rate',type = float)
parser.add_argument('-m','--momentum',type = float)
parser.add_argument('-beta','--beta',type = float)
parser.add_argument('-beta1','--beta1',type = float)
parser.add_argument('-beta2','--beta2',type = float)
parser.add_argument('-eps','--epsilon',type = float)
parser.add_argument('-w_d','--weight_decay',type = float)
parser.add_argument('-w_i','--weight_init')
parser.add_argument('-nhl','--num_layers',type = int)
parser.add_argument('-sz','--hidden_size',type = int)
parser.add_argument('-a','--activation')

args = parser.parse_args()

def main():
    '''
    default values'''
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

    if args.wandb_project is not None:
        project_name=args.wandb_project
    if args.wandb_entity is not None:
        entity_name=args.wandb_entity
    if args.dataset is not None:
        dataset_name=args.dataset
    if args.learning_rate is not None:
        learning_rate=args.learning_rate
    if args.momentum is not None:
        mom=args.momentum
    if args.beta is not None:
        beta=args.beta
    if args.beta1 is not None:
        beta1=args.beta1
    if args.beta2 is not None:
        beta2=args.beta2
    if args.epsilon is not None:
        epsilon=args.epsilon
    if args.optimizer is not None:
        optimizer=args.optimizer
    if args.batch_size is not None:
        batch_size=args.batch_size
    if args.loss is not None:
        loss=args.loss
    if args.epochs is not None:
        epochs=args.epochs
    if args.weight_decay is not None:
        wd=args.weight_decay
    if args.weight_init is not None:
        wi=args.weight_init
    if args.num_layers is not None:
        hls=args.num_layers
    if args.hidden_size is not None:
        nhls=args.hidden_size
    if args.activation is not None:
        act=args.activation
    

    if(dataset_name=="fashion_mnist"):
        (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    else:
        (x_train,y_train),(x_test,y_test)=mnist.load_data()
    validation_ratio=0.1
    num_validation_samples=int(validation_ratio*x_train.shape[0])
    validation_indices=np.random.choice(x_train.shape[0],num_validation_samples,replace=False)

    x_val,y_val=x_train[validation_indices],y_train[validation_indices]
    x_train, y_train=np.delete(x_train,validation_indices,axis=0),np.delete(y_train,validation_indices,axis=0)

    
    wandb.init(project=project_name,entity=entity_name)
    Model=FeedForwardNeuralNetwork(x_train,y_train,x_val,y_val,x_test,y_test,hls=hls,neurons_in_hl=nhls,activation=act,initialization=wi,epochs=epochs,eta=learning_rate,wd=wd,loss=loss)
    Model.modelFitting(mom=mom,beta=beta,beta1=beta1,beta2=beta2,eps=epsilon,optimizer=optimizer,batch_size=batch_size)

if __name__ == '__main__':
    main()