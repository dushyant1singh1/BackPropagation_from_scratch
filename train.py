import argparse
from feedForward import FeedForwardNN
from tensorflow.keras.datasets import fashion_mnist,mnist
import wandb
import numpy as np
def load_data(name):
        if(name=='fashion_mnist'):
                (train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()
                x_train,y_train,x_val,y_val= data_processing(train_x,train_y)
                return x_train,y_train,x_val,y_val,test_x,test_y                


        elif(name=='mnist'):
                (train_X, train_y), (test_X, test_y) = mnist.load_data()
                x_train,y_train,x_val,y_val = data_processing(train_x,train_y)
                return X_train,y_train,x_val,y_val,test_x,test_y

def data_processing(train_x,train_y):
        split = 0.1
        total_data = train_x.shape[0]
        indices = np.arange(total_data)
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        data_train = int((1-split)*total_data)
        x_train = train_x[:data_train]
        y_train = train_y[:data_train]
        x_val = train_x[data_train:]
        y_val = train_y[data_train:]
        return x_train,y_train,x_val,y_val



parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",default="myprojectname")
parser.add_argument("-we","--wandb_entity",default="myname")
parser.add_argument("--dataset", "-d", help = "dataset", 
choices=["mnist","fashion_mnist"], default="fashion_mnist")
parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
type= int, default=10)
parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
, type =int, default=16)
parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
default= "sgd", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
parser.add_argument("--loss","-l",choices=["mean_squared_error", "cross_entropy"],default="cross_entropy")
parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
parser.add_argument("--momentum","-m",default=0.5,type=float)
parser.add_argument("--beta","-beta",default=0.5, type=float)
parser.add_argument("--beta1","-beta1",default=0.5,type=float)
parser.add_argument("--beta2","-beta2",default=0.5,type=float)
parser.add_argument("--epsilon","-eps",type=float,default = 0.000001)
parser.add_argument("--weight_decay","-w_d",default=0.0,type=float)
parser.add_argument("-w","--weight_init",default="random",choices=["random","Xavier"])
parser.add_argument("--num_layers","-nhl",type=int,default=1)
parser.add_argument("--hidden_size","-sz",type=int,default=4)
parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"],default="sigmoid")
#parser.add_argument()

args = parser.parse_args()
# print(args.dataset)
# print(args.epochs)
# print(args.batch_size)
# print(args.optimizer)
# print(args.loss)
# print(args.learning_rate)
# print(args.momentum)
# print(args.beta)
# print(args.beta1)
# print(args.beta2)
# print(args.epsilon)
# print(args.weight_decay)
# print(args.weight_init)
# print(args.num_layers)
# print(args.hidden_size)
# print(args.activation)
train_x,train_y,x_val,y_val,test_x,test_y = load_data(args.dataset)
classes  = 10
# wandb.login()
# wandb.init()
sizeHL = [args.hidden_size for i in range(args.num_layers)]
ob = FeedForwardNN(args.num_layers,sizeHL,train_x,train_y,x_val,y_val,test_x,test_y,args.optimizer,args.batch_size,args.learning_rate,args.epochs,args.activation,args.weight_init,args.loss, args.weight_decay)
ob.train(optimizer=args.optimizer,beta1=args.beta1,beta2=args.beta2)




        


