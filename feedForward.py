import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tqdm import tqdm
import decimal

def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1.0 / (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)

def relu(z):
    return (z>0)*(z) + ((z<0)*(z)*0.01)
    #return np.maximum(z,0)
    #return np.where(z<0, 0.01*z, z)

# def softmax(Z):
#     ep = 1e-5
#     Z = np.clip(Z,-600,600)
#     return (np.exp(Z) / (np.sum(np.exp(Z))))

def softmax(Z):
    c = Z.max()
    logsumexp = np.log(np.exp(Z - c).sum())
    return Z - c - logsumexp


def grad_sigmoid(z):
    return  sigmoid(z)*(1-sigmoid(z))

def grad_tanh(z):
    return 1 - np.tanh(z) ** 2


def grad_relu(z):
    return (z>0)*np.ones(z.shape) + (z<0)*(0.01*np.ones(z.shape) )


class FeedForwardNN:
    def __init__(
        self, layers, sizeHL, x_train, y_train, 
        x_test, y_test, optimizer, batchSize, weightDecay, lr, iterations, activation,
        initializer, loss):
        self.classes = 10
        self.layers = layers+2
        self.layersSize = [784] + sizeHL + [10]
        self.total = x_train.shape[0]
        # 784 * 60000
        self.X = np.transpose(x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        self.X_test = np.transpose(x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
        #normalizing values
        self.X = self.X/255
        self.X_test = self.X_test/255
        # changing normal values to one hot encoding
        # shape will be 10*60000
        
        self.Y = self.oneHotEncoder(y_train)
        self.Y_test = self.oneHotEncoder(y_test)
        self.y_train = y_train
        self.Activations_dict = {"SIGMOID": sigmoid, "TANH": tanh, "RELU": relu}
        self.derActivation_dict = {
            "SIGMOID": grad_sigmoid,
            "TANH": grad_tanh,
            "RELU": grad_relu,
        }

        self.Initializer_dict = {
            "XAVIER": self.xavier,
            "RANDOM": self.random,
        }

        # self.Optimizer_dict = {
        #     "SGD": self.sgd,
        #     "MGD": self.mgd,
        #     "NAG": self.nag,
        #     "RMSPROP": self.rmsProp,
        #     "ADAM": self.adam,
        #     "NADAM": self.nadam,
        # }
        self.activation = self.Activations_dict[activation]
        self.derivation_activation = self.derActivation_dict[activation]
    #    self.optimzer = self.Optimizer_dict[optimizer]
        self.initializer = self.Initializer_dict[initializer]
        self.lossFunction = loss
        self.epochs = iterations
        self.batchSize = batchSize
        self.lr = lr

        # initializing weights and biases
        self.weights, self.biases = self.initializeWeights(self.layersSize)

    def oneHotEncoder(self,rawY):
        onehot = np.zeros((self.classes,rawY.shape[0]))
        size = rawY.shape[0]
        for i in range(size):
            onehot[int(rawY[i])][i] = 1.0
        return onehot
        

    def accuracy(self):
        pred,H,A = self.forwardPropagation(self.X)
        pred = pred.T
        count =0
        for i in range(self.total):
            if(np.argmax(pred[i])==self.y_train[i]):
                count+=1
        return count/(self.total)

    def initializeWeights(self,layersSize):
        weights =[]
        biases =[]
        for i in range(len(layersSize)-1):
            weights.append(self.initializer([layersSize[i+1],layersSize[i]]))
            biases.append(self.initializer([layersSize[i+1],1]))
        return weights,biases

    def xavier(self, size):
        std = np.sqrt(2 / (size[0] + size[1]))
        # size[0] = next layer's neurons
        # size[1] = prev layer's neurons
        return np.random.normal(0, std, size=(size[0], size[1]))

    def random(self, size):
        return np.random.normal(0, 1, size=(size[0], size[1]))

    def forwardPropagation(self,X):
        layers = len(self.weights)
        H =[0 for i in range(layers-1)]
        A =[0 for i in range(layers)]
        for i in range(layers-1):
            if i==0:
                A[i] = np.add(np.matmul(self.weights[i],X),self.biases[i])
                H[i] = self.activation(A[i])
            else:
                A[i] = np.add(np.matmul(self.weights[i],H[i-1]),self.biases[i])
                H[i] = self.activation(A[i])
        A[layers-1] = np.add(np.matmul(self.weights[layers-1],H[layers-2]),self.biases[layers-1])
        #print(A[layers-1])
        pred = softmax(A[layers-1])
        return pred,H,A

    def backPropagation(self, pred, H, A, x_train, y_train, weight_decay=0 ):
        dW = []
        dB = []
        # x_train has the shape of 784*batch size
        #output layer gradient
        #if self.lossFunction =='CROSS':
        gradients ={}
        l = len(self.layersSize)-2
        # print(l)
        if self.lossFunction =='CROSS':
            gradients['a'+str(l)] = -(y_train - pred)
        elif self.lossFunction=='MSE':
            gradients['a'+str(l)] = np.multiply(2*(pred-y_train),np.multiply(pred,(1-pred))) 
        
        # print(l)
        for i in range(l,0,-1):
            dw = np.dot(gradients['a'+str(i)],H[i-1].T)
            db = np.sum(gradients['a'+str(i)],axis=1).reshape(-1,1)
            dW.append(dw)
            dB.append(db)
            # print("iteration : {i}")
            # print(self.weights[i].shape)
            # print(gradients['a'+str(i)].shape)
            dh = np.matmul(self.weights[i].T,gradients['a'+str(i)])
            gradients['a'+str(i-1)] = np.multiply(dh,self.derivation_activation(A[i-1]))
        dW.append(np.dot(gradients['a'+str(0)],x_train.T))
        dB.append(np.sum(gradients['a'+str(0)],axis =1).reshape(-1,1))
        dW.reverse()
        dB.reverse()
        return dW,dB
    
    def sgd(self,weight_decay=0):
        iterations = self.epochs
        layers = self.layers

        totalData = self.X.shape[-1]
        for i in tqdm(range(iterations)):
            j =0
            dW=[]
            dB=[]
            while(j<totalData):
                pred,H,A  = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                dW,dB = self.backPropagation(pred, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])

                j+=self.batchSize

                for k in range(layers-1):
                    self.weights[k] = self.weights[k] - self.lr*dW[k]
                    self.biases[k] = self.biases[k] - self.lr*dB[k]
            print("Accuracy for %d iteration is %.5f"%(i+1,self.accuracy()))
            
    def mgd(self,beta,weight_decay=0):
        print(self.derivation_activation)
        iterations = self.epochs
        layers = self.layers
        uW = [0 for i in range(layers-1)]
        uB = [0 for i in range(layers-1)]
        totalsamples = self.X.shape[-1]
        for i in tqdm(range(iterations)):
            j =0
            dW =[]
            dB =[]
            while(j<totalsamples):
                pred, H, A = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                dW, dB = self.backPropagation(pred, H, A, self.X[:,j:j+batchSize],self.Y[:,j:j+batchSize])
                j+=self.batchSize

                for k in range(layers-1):
                    uW[k] = uW[k]*beta + dW[k]
                    uB[k] = uB[k]*beta + dB[k]
                
                for k in range(layers-1):
                    self.weights[k] -= self.lr*uW[k]
                    self.biases[k] -= self.lr*uB[k]

            print("Accuracy for %d iteration is %.5f"%(i+1,self.accuracy()))
            






(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()
classes =['Ankle boot','T-shirt/top','Dress','Pullover','sneaker','Sandal','Trouser','Shirt','Coat','Bag']


layers =3
sizeHL =[128,128,128]
optimizer = 'SGD'
batchSize = 32
weightDecay = 0
lr = 0.0001
iterations = 5
activation = 'TANH'
initializer = 'RANDOM'
loss= 'CROSS'
momentum = 0.9
# layers, sizeHL, x_train, y_train, x_test, y_test, optimizer, batchSize, weightDecay, lr, iterations, activation,initializer, loss
ob = FeedForwardNN(3,sizeHL,train_x,train_y,test_x,test_y,'SGD',batchSize,weightDecay,lr,iterations,activation,initializer,loss)
ob.mgd(momentum)