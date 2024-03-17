import argparse
#from feedForward import FeedForwardNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help = "dataset", 
    choices=["mnist","fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
            type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
    , type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
     default= "sgd", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l",choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--momentum","-m",default=0.5,type=float)
    parser.add_argument("--beta","-beta",default=0.5, type=float)
    parser.add_argument("--beta1","-beta1",default=0.5,type=float)
    parser.add_argument("--beta2","-beta2",default=0.5,type=float)
    parser.add_argument("--epsilon","-eps",type=float,default = 0.000001)
    parser.add_argument("--weight_decay","-w_d",default=0.0,type=float)
    parser.add_argument("-w","--weight_init",default="random",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int,default=1)
    parser.add_argument("--hidden_size","-sz",type=int,default=4)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"],default="sigmoid")
    #parser.add_argument()
    
    
    
    args = parser.parse_args()
    print(args.dataset)
    print(args.epochs)
    print(args.batch_size)
    print(args.optimizer)
    print(args.loss)
    print(args.learning_rate)
    print(args.momentum)
    print(args.beta)
    print(args.beta1)
    print(args.beta2)
    print(args.epsilon)
    print(args.weight_decay)
    print(args.weight_init)
    print(args.num_layers)
    print(args.hidden_size)
    print(args.activation)

