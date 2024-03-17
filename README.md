# CS6910 Assignment 1

## Getting the code files
You need to first clone the github repository containing the files.
```
git clone https://github.com/pritamde090920/cs6910_assignment1.git
```
Then change into the code directory.
```
cd cs6910_assignment1
```
Make sure you are in the correct directory before proceeding further.

## Setting up the platform and environment
- ### Local machine
  If you are running the code on a local machine, then you need to have ython installed in the machine and pip command added in the environemnt variables.
  The following modules need to be installed :
  - [numpy](https://numpy.org/doc/)
  - [wandb](https://docs.wandb.ai/)
  - [tensorflow](https://www.tensorflow.org/guide)
  - [keras](https://keras.io/guides/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [matplotlib](https://matplotlib.org/stable/index.html)
  
  A module can be installed by using the following command through command prompt : ```pip install <module_name>```.\
  Once all the packages are installed, then the codes can be excuted from any python IDE like [VSCode](https://code.visualstudio.com/docs), [PyCharm](https://www.jetbrains.com/help/pycharm/getting-started.html), [IDLE](https://docs.python.org/3/library/idle.html), [Jupyter Notebook](https://docs.jupyter.org/en/latest/)etc.

- ### Google colab/Kaggle
  If you are using google colab platform or kaggle, then you don't need to do anything. You can directly execute the codes.

## Training the model
To train the model, you need to compile and execute the [train.py](https://github.com/pritamde090920/cs6910_assignment1/blob/main/train.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train.py
```
By the above command, the model will run with the default configuration.\
To customize the run, you need to specify the parameters like ```python train.py <*args>```\
For example,
```
python train.py -d mnist -b 32 -e 5 --optimizer nag 
```
The arguments supported are : 

|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Pritam CS6910 - Assignment 1 | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |   cs23m051    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|     `-d`, `--dataset`    | fashion_mnist | choices:  ["mnist", "fashion_mnist"]                                      |
|     `-e`, `--epochs`     |       30      | Number of epochs to train neural network.                                 |
|   `-b`, `--batch_size`   |       32      | Batch size used to train neural network.                                  |
|      `-l`, `--loss`      | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"]                         |
|    `-o`, `--optimizer`   |      nadam    | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]          |
| `-lr`, `--learning_rate` |      1e-3     | Learning rate used to optimize model parameters                           |
|    `-m`, `--momentum`    |      0.9      | Momentum used by momentum and nag optimizers.                             |
|     `-beta`, `--beta`    |      0.9      | Beta used by rmsprop optimizer                                            |
|    `-beta1`, `--beta1`   |      0.9      | Beta1 used by adam and nadam optimizers.                                  |
|    `-beta2`, `--beta2`   |      0.999    | Beta2 used by adam and nadam optimizers.                                  |
|    `-eps`, `--epsilon`   |      1e-5     | Epsilon used by optimizers.                                               |
| `-w_d`, `--weight_decay` |       .0      | Weight decay used by optimizers.                                          |
|  `-w_i`, `--weight_init` |     Xavier    | choices:  ["random", "Xavier"]                                            |
|  `-nhl`, `--num_layers`  |       4       | Number of hidden layers used in feedforward neural network.               |
|  `-sz`, `--hidden_size`  |       64      | Number of hidden neurons in a feedforward layer.                          |
|   `-a`, `--activation`   |     ReLU      | choices:  ["identity", "sigmoid", "tanh", "ReLU"]                         |
|   `-c`,`--confusion`     |       0       | Generate confusion matrix. choices:  [0,1]                                |
|    `-t`,`--test`         |       0       | Generate test accuracy. choices:  [0,1]                                   |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the key in the interactive command prompt.

## Testing the model
To test the model, you need to execute the train.py and set the test flag as 1 in the command line. If you want to test it on mnist dataset, then set dataset as mnist.
```
python train.py -t 1 -d mnist
```
This will run the model with default parameters in mnist dataset and print the test accuracy.

## Additional features
The following features are also supported
  - If you need some clarification on the arguments to be passed, then you can do
    ```
    python train.py --help
    ```
  - If you want to see the confusion matrix of the test dataset, then you can do
    ```
    python train.py -confusion 1
    ```
    This will make a log of the confusion matrix generated by the model on the dataset into wandb dashboard

## Links
[Wandb Report](https://wandb.ai/cs23m051/Pritam%20CS6910%20-%20Assignment%201/reports/CS6910-Assignment-1-Pritam-De-CS23M051--Vmlldzo3MDkyOTY2)
