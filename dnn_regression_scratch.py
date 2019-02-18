import numpy as np
import math


def sigmoid(x):
    #r= 1 / (1 + math.exp(-x)) => sigmoid function
    # faster to use np.exp (which will be applied to each element in the array) than math.exp
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(fx):
    return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(y.shape)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def cost(self, dataset,labels):
        error = 0.0
        for xi,yi in zip(dataset, labels) :
            self.input = xi
            self.y = yi
            self.feedforward()
            error += math.pow((self.y - self.output),2)
        return error

x = np.random.rand(1,3)
y = np.random.rand(1)
#print(x,y)
#model = NeuralNetwork(x,y)
#print(model, model.weights1, model.weights2, model.output)

from scipy import io as spio
from mlxtend.data import loadlocal_mnist

x, y = loadlocal_mnist(
        images_path='../MNIST Data Set/train-images.idx3-ubyte',
        labels_path='../MNIST Data Set/train-labels.idx1-ubyte') ## data set downloaded from 'http://yann.lecun.com/exdb/mnist/'

# print(x[0],y[0])
# shape x = 60000 * 784 = 60000 training example/input, 784 features
epoch = 5
def training(trainset, labels):
    print(np.array(trainset[0]).T.shape)
    model = NeuralNetwork(np.reshape(trainset[0],(1,784)), np.array([labels[0]]))
    for i in range(epoch):
        for xi,yi in zip(trainset, labels) :
            model.input = np.reshape(xi, (1,784))
            model.y = np.array([yi])
            model.feedforward()
            model.backprop()

        print("for this iteration error =  ",model.cost(trainset,labels))

# i = x[0].shape[0]
# z = np.array(x[0])
# print(z)
# z.reshape(1,784)
# B = np.reshape(z, (1, 784))
# z.shape += (1,)
# print(B.shape)
# print(B)
# print(x[0].shape, z.T.shape)
# w = np.array([y[0]])
# print(w.shape)
a = np.reshape(x[0],(28,28))
print(a)

training(x,y)

