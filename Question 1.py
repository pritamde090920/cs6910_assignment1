import wandb
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.datasets import fashion_mnist

'''login to wandb to generate plot'''
wandb.login()

'''loads dataset from fashion_mnist'''
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

'''gives labels of the 10 output classes'''
output_class=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

'''initialize the project'''
wandb.init(project="Pritam CS6910 - Assignment 1",name="Question 1")

img,getPlot=plt.subplots(2,5,figsize=(20,6))
getPlot=getPlot.flatten()
output_images=[]
for i in range(10):
  '''get the index where the image of the particular class is present'''
  imgClass=np.argmax(y_train==i)
  '''get the image pixels'''
  getPlot[i].imshow(x_train[imgClass],cmap="gray")
  getPlot[i].set_title(output_class[i])
  '''generate plot for each class'''
  img=wandb.Image(x_train[imgClass],caption=[output_class[i]])
  output_images.append(img)
'''log the images into wandb'''
wandb.log({"Question 1":output_images})
wandb.finish()