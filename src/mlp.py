import numpy as np

class MLP:
    layers_list = list()

    def __init__(self, layers_sizes, ):
        for i in range(len(layers_sizes)-1):
            self.layers_list.append(Layer(layers_sizes[i], layers_sizes[i+1]))
            # adds layer objects into array with parameters of the number of a) input neurons and b) neurons in layer
            # one less layer than size of layers because first element of layers_size array is input, not layer



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

class Layer:
    W = None
    # matrix where each row indicates the list of weights that are inputs for a particular node
    b = None
    # vector that indicates biases for every node in the layer
    z = None
    # cached result from forward pass that allows to calculate gradients in backward pass
    # z is the input into the activation function
    dW = None
    db = None
    inputs = None
    # cached input from the parameter passed to the forward pass
    # vector of inputs to a layer from previous layer

    def __init__(self, input_size, output_size, scale):
        self.W = np.random.randn((output_size, input_size)) * scale
        # random so network can actually learn, otherwise weights would change by same amount for all nodes
        self.b = np.zeros((output_size, 1)) 
        self.dW = np.zeros(output_size, input_size)
        self.db = np.zeros(output_size, 1)

    def forward_pass(self, x):
        self.z = np.dot(self.W,x) + self.b
        self.inputs = x
        
        return sigmoid(self.z)
        # returns

    def backward_pass(self, dOutput):
        dw = np.dot((dOutput * sigmoid_derivative(self.z)), self.inputs.T)
        # dw found by multiplying dOutput by sigmoid derivative of z and then by input.
        # doing dot product this way round with transposed vector second produces a matrix rather than number
        # since the column vector here is the output-node-related value, each row of resulting matrix corresponds to one node in the current (output) layer, just like the W matrix

        db = dOutput * sigmoid_derivative(self.z)
        # dz/db = 1 

        return (dOutput * sigmoid_derivative(self.z)) * self.W
        # every row in W corresponds to weight inputs to a node in the output layer (the one belonging tho this object)
        # uses broadcasting to multiply every weight in the row by the corresponding value for the output node in the column vector
        # returns dInput (eventually passed as input to backprop of previous layer)
        
