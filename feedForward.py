import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tqdm import tqdm
import decimal

def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1.0 / (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)

# def relu(z):
#     return (z>0)*(z) + ((z<0)*(z)*0.01)
#     #return np.maximum(z,0)
#     #return np.where(z<0, 0.01*z, z)
def relu(z):
    return np.maximum(z*0.01,z)


def softmax(Z):
    ep = 1e-5
    Z = np.clip(Z,-600,600)
    return (np.exp(Z) / (np.sum(np.exp(Z)))+ep)

# def softmax(Z):
#     c = Z.max()
#     logsumexp = np.log(np.exp(Z - c).sum())
#     return Z - c - logsumexp


def grad_sigmoid(z):
    return  sigmoid(z)*(1-sigmoid(z))

def grad_tanh(z):
    return 1 - np.tanh(z) ** 2


# def grad_relu(z):
#     return (z>0)*np.ones(z.shape) + (z<0)*(0.01*np.ones(z.shape) )

def grad_relu(x):
    alpha=0.01
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

class FeedForwardNN:
    def __init__(
        self, layers, sizeHL, x_train, y_train, x_val, y_val, 
        x_test, y_test, optimizer, batchSize, weightDecay, lr, iterations, activation,
        initializer, loss):
        self.classes = 10
        self.layers = layers+2
        self.layersSize = [784] + sizeHL + [10]
        self.total = x_train.shape[0]
        # 784 * 60000
        self.X = np.transpose(x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        self.X_test = np.transpose(x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
        self.X_val = np.transpose(x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2]))
        #normalizing values
        self.X = self.X/255
        self.X_test = self.X_test/255
        self.X_val = self.X_val/255
        # changing normal values to one hot encoding
        # shape will be 10*60000
        
        self.Y = self.oneHotEncoder(y_train)
        self.Y_test = self.oneHotEncoder(y_test)
        self.Y_val = self.oneHotEncoder(y_val)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
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
        

    def accuracy(self,x,y):
        pred,H,A = self.forwardPropagation(x)
        pred = pred.T
        count =0
        for i in range(len(y)):
            if(np.argmax(pred[i])==y[i]):
                count+=1
        return count/(len(y))

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
                A[i] = np.add(np.dot(self.weights[i],X),self.biases[i])
                H[i] = self.activation(A[i])
            else:
                A[i] = np.add(np.dot(self.weights[i],H[i-1]),self.biases[i])
                H[i] = self.activation(A[i])
        A[layers-1] = np.add(np.dot(self.weights[layers-1],H[layers-2]),self.biases[layers-1])
        #print(A[layers-1])
        pred = softmax(A[layers-1])
        return pred,H,A

    def backPropagation(self, pred, weights, H, A, x_train, y_train, weight_decay=0 ):
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
        #print(weights[0])
        # print(l)
        for i in range(l,0,-1):
            dw = np.dot(gradients['a'+str(i)],H[i-1].T)
            db = np.sum(gradients['a'+str(i)],axis=1).reshape(-1,1)
            dW.append(dw)
            dB.append(db)
            # print("iteration : {i}")
            # print(self.weights[i].shape)
            # print(gradients['a'+str(i)].shape)
            
            dh = np.matmul(weights[i].T,gradients['a'+str(i)])
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
                dW,dB = self.backPropagation(pred, self.weights, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])

                j+=self.batchSize

                for k in range(layers-1):
                    self.weights[k] = self.weights[k] - self.lr*dW[k]
                    self.biases[k] = self.biases[k] - self.lr*dB[k]
            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))
            
    def mgd(self,beta,weight_decay=0):
        #print(self.derivation_activation)
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
                dW, dB = self.backPropagation(pred, self.weights, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])
                j+=self.batchSize

                for k in range(layers-1):
                    uW[k] = uW[k]*beta + dW[k]
                    uB[k] = uB[k]*beta + dB[k]
                
                for k in range(layers-1):
                    self.weights[k] -= self.lr*uW[k]
                    self.biases[k] -= self.lr*uB[k]

            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))
            
    def nag(self,beta):
        iterations = self.epochs
        layers = self.layers
        print(layers)
        vW = [0 for i in range(layers-1)]
        vB = [0 for i in range(layers-1)]
        pvW = [0 for i in range(layers-1)]
        pvB = [0 for i in range(layers-1)]
        v_W = [0 for i in range(layers-1)]
        v_B = [0 for i in range(layers-1)]
        totalpoints = self.X.shape[-1]
        for i in tqdm(range(iterations)):
            j =0
            dW =[]
            dB =[]
            for k in range(layers-1):
                v_W[k] = beta*pvW[k]
                v_B[k] = beta*pvB[k]
            while(j<totalpoints):
                nw=[]
                pred, H, A = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                for p in range(layers-1):
                    nw.append(self.weights[p] - v_W[p])
                dW,dB = self.backPropagation(pred, nw, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])

                for l in range(layers-1):
                    vW[l] = beta*pvW[l] + self.lr*dW[l]
                    self.weights[l] -= vW[l]
                    vB[l] = beta*pvB[l] + self.lr*dB[l]
                    self.biases[l] -= vB[l]
                pvW = vW
                pvB = vB
                j+=self.batchSize
            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))

    def rmsProp(self,beta,weight_decay=0):
        
        layers = self.layers                    
        print(layers)
        vW = [0 for i in range(layers-1)]
        vB = [0 for i in range(layers-1)]
        eps =1e-4
        totalsamples = self.X.shape[-1]
        for i in tqdm(range(self.epochs)):
            j =0
            while(j<totalsamples):
                pred, H, A = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                dW, dB = self.backPropagation(pred, self.weights,H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])


                for k in range(layers-1):
                    vW[k] = beta*vW[k] + (1-beta)*(dW[k]**2)
                    vB[k] = beta*vB[k] + (1-beta)*(dB[k]**2)
                for k in range(layers-1):
                    self.weights[k] -= self.lr*(dW[k]/(np.sqrt(vW[k])+eps))
                    self.biases[k] -= self.lr*(dB[k]/(np.sqrt(vB[k])+eps))
                
                j+=self.batchSize
            
            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))
                

    def adam(self,beta1,beta2):
        layers = self.layers
        m_w,m_b,v_w,v_b = [[0 for l in range(layers-1)] for k in range(4)]
        totalsamples = self.X.shape[-1]
        eps = 1e-10
        for i in range(self.epochs):
            j =0

            while(j<totalsamples):
                pred, H, A = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                dW, dB = self.backPropagation(pred, self.weights, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])

                j+=self.batchSize

                for p in range(layers-1):
                    m_w[p] = beta1*m_w[p] + (1-beta1)*dW[p]
                    m_b[p] = beta1*m_b[p] + (1-beta1)*dB[p]
                    v_w[p] = beta2*v_w[p] + (1-beta2)*(dW[p]**2)
                    v_b[p] = beta2*v_b[p] + (1-beta2)*(dB[p]**2)
                m_w_hat,m_b_hat,v_b_hat,v_w_hat = [[0 for c in range(layers-1)] for q in range(4)]
                for c in range(layers-1):
                    m_w_hat[c] = m_w[c]/(1-np.power(beta1,i+1))
                    m_b_hat[c] = m_b[c]/(1-np.power(beta1,i+1))
                    v_w_hat[c] = v_w[c]/(1-np.power(beta2,i+1))
                    v_b_hat[c] = v_b[c]/(1-np.power(beta2,i+1))

                for c in range(layers-1):
                    self.weights[c] -= self.lr*m_w_hat[c]/(np.sqrt(v_w_hat[c])+eps)
                    self.biases[c] -= self.lr*m_b_hat[c]/(np.sqrt(v_b_hat[c])+eps)
            
            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))

    def nadam(self,beta1,beta2):
        layers = self.layers
        m_w,m_b,v_w,v_b = [[0 for i in range(layers-1)] for k in range(4)]
        eps = 1e-10
        totalsamples = self.X.shape[-1]
        for i in range(self.epochs):
            j = 0
            while(j<totalsamples):
                pred, H, A = self.forwardPropagation(self.X[:,j:j+self.batchSize])
                dW, dB = self.backPropagation(pred, self.weights, H, A, self.X[:,j:j+self.batchSize],self.Y[:,j:j+self.batchSize])
                j+=self.batchSize

                for p in range(layers-1):            
                    m_w[p] = beta1*m_w[p]+(1-beta1)*dW[p]
                    m_b[p] = beta1*m_b[p]+(1-beta1)*dB[p]
                    v_w[p] = beta2*v_w[p]+(1-beta2)*dW[p]**2
                    v_b[p] = beta2*v_b[p]+(1-beta2)*dB[p]**2

                m_w_hat,m_b_hat,v_b_hat,v_w_hat = [[0 for c in range(layers-1)] for q in range(4)]
                for c in range(layers-1):
                    m_w_hat[c] = m_w[c]/(1-np.power(beta1,i+1))
                    m_b_hat[c] = m_b[c]/(1-np.power(beta1,i+1))
                    v_w_hat[c] = v_w[c]/(1-np.power(beta2,i+1))
                    v_b_hat[c] = v_b[c]/(1-np.power(beta2,i+1))
                    
                for c in range(layers-1):
                    self.weights[c] -= (self.lr/np.sqrt(v_w_hat[c]+eps))*(beta1*m_w_hat[c]+(1-beta1)*dW[c]/(1-beta1**(i+1)))
                    self.biases[c] -= (self.lr/np.sqrt(v_b_hat[c]+eps))*(beta1*m_b_hat[c]+(1- beta1)*dB[c]/(1-beta1**(i+1)))
                
            
            print("Train Accuracy - %.5f, Val Accuracy - %.5f EPOCH ==> %d"%(self.accuracy(self.X,self.y_train),self.accuracy(self.X_val,self.y_val),i+1))


(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()
classes =['Ankle boot','T-shirt/top','Dress','Pullover','sneaker','Sandal','Trouser','Shirt','Coat','Bag']


layers = 4
sizeHL =[512,128,64,32]
optimizer = 'SGD'
batchSize = 200
weightDecay = 0
lr = 0.0001
iterations = 100
activation = 'TANH'
initializer = 'XAVIER'
loss= 'CROSS'
momentum = 0.4
beta1 = 0.5
beta2 = 0.5
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

print(x_train.shape,y_train.shape)
print(x_val.shape,y_val.shape)
# layers, sizeHL, x_train, y_train, x_val, y_val, x_test, y_test, optimizer, batchSize, weightDecay, lr, iterations, activation,initializer, loss
ob = FeedForwardNN(layers,sizeHL,x_train,y_train,x_val,y_val,test_x,test_y,'SGD',batchSize,weightDecay,lr,iterations,activation,initializer,loss)
ob.nadam(beta1,beta2)