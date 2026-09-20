import numpy as np
from sklearn.datasets import make_moons

# NOTES: 
# this script uses the column convention so that each training example in a batch is in one column
# this includes X and y inputs and any derived values.

config = {
    "learning_rate": 0.01, # dictates how fast weights adjust from gradient descent
    "batch_size": 5, # number of training examples used in one forward and backward pass
    "num_epochs": 20, # 
    "layers_sizes": [2, 16, 8, 1], # array of the number of neurons in each layer, that is unpacked later
    "scale": 0.01, # the scalar that the randomly generated weights are multiplied by at the start
    "num_samples": 1000, # how many traning examples are even retrieved from the dataset
    "noise": 0.2
}


def main():
    model = MLP(config["layers_sizes"], config["scale"])
    

    
def get_batches(X, y, batch_size):
    num_samples = X.shape[1]
    # this is the number of total training examples
    random_indices = np.random.permutation(X.shape[1])
    # X is a matrix that represents all the inputs (each column is a training exanmple to correspond to each column of y)
    X_shuffled = X[:, random_indices]
    y_shuffled = y[:, random_indices]
    # the X and y are shuffled beforehand for stochastic gradient descent, so that later when picking batches we can be sure that every training example is trained upon
    for start_index in range(0, num_samples, batch_size):
        yield X_shuffled[:, start_index:(start_index+batch_size)], y_shuffled[:, start_index : (start_index + batch_size)]

def train_model(model, X, y, config):


    for epoch in range(config["num_epochs"]):
        # the number of epochs decides how many cycles there are through the FULL dataset
        epoch_losses = []

        for X_batch, y_batch in get_batches(X, y, config["batch_size"]):
            yhat = model.forward_pass(X_batch)
            L = (1/(2*config["batch_size"])) * mse(yhat, y_batch)
            epoch_losses.append(L)
            # to track the average losses for each epoch
            dyhat = (1 / config["batch_size"] ) * mse_derivative(yhat, y_batch)
            model.backward_pass(dyhat)
            model.update_parameters(config["learning_rate"])

        print(f"Epoch {epoch}: {np.mean(epoch_losses)}")





                

class MLP:
    
    def __init__(self, layer_sizes, scale):
        self.layers_list = []
        for i in range(len(layer_sizes)-1):
            self.layers_list.append(Layer(layer_sizes[i], layer_sizes[i+1], scale))
            # adds layer objects into array with parameters of the number of a) input neurons and b) neurons in layer
            # one less layer than size of layers because first element of layers_size array is input, not layer

    def forward_pass(self, x):
        A = x
        for layer in self.layers_list:
            A = layer.forward_pass(A)
        return A

    def backward_pass(self, dOutput):
        for i in range(len(self.layers_list)):
            dOutput = self.layers_list[-i-1].backward_pass(dOutput)
        pass

    def update_parameters(self, learning_rate):
        for layer in self.layers_list:
            layer.update_parameters(learning_rate)
        
        




    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def mse(yhat, y):
    return np.sum((yhat-y) ** 2)

def mse_derivative(yhat, y):
    return (yhat - y)

class Layer:
    # all inputs and outputs variables are matrices because each back and forward pass uses <batch_size> training examples
    

    def __init__(self, input_size, output_size, scale):
        self.W = np.random.randn(output_size, input_size) * scale
        # matrix where each row indicates the list of weights that are inputs for a particular node
        # random so network can actually learn, otherwise weights would change by same amount for all nodes
        self.b = np.zeros((output_size, 1)) 
        # vector that indicates biases for every node in the layer
        self.dW = np.zeros((output_size, input_size))
        self.db = np.zeros((output_size, 1))
        self.inputs = None
        # input from the parameter passed to the forward pass to be cached later
        # vector of inputs to a layer from previous layer
        self.Z = None
        # cached result from forward pass that allows to calculate gradients in backward pass
        # Z is the input into the activation function
        

    def forward_pass(self, X):
        self.Z = np.dot(self.W,X) + self.b
        self.inputs = X
       
        return sigmoid(self.Z)
        # returns

    def backward_pass(self, dOutput):
        self.dW = np.dot((dOutput * sigmoid_derivative(self.Z)), self.inputs.T)
        # dw found by multiplying dOutput by sigmoid derivative of z and then by input.
        # doing dot product this way round with transposed vector second produces a matrix rather than number
        # since the column vector here is the output-node-related value, each row of resulting matrix corresponds to one node in the current (output) layer, just like the W matrix

        self.db = dOutput * sigmoid_derivative(self.Z)
        # dZ/db = 1 
        dZ = (dOutput * sigmoid_derivative(self.Z))
        return np.dot(self.W.T, dZ)
        # every row in W corresponds to weight inputs to a node in the output layer (the one belonging tho this object)
        # uses broadcasting to multiply every weight in the row by the corresponding value for the output node in the column vector
        # returns dInput (eventually passed as input to backprop of previous layer)

    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
