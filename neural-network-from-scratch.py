import numpy as np
import matplotlib.pyplot as plt

import Q1_dataset
np.random.seed(0)

training_size = 25

generated_matrix = Q1_dataset.generated_matrix

data = [i[0] for i in generated_matrix]
classes = [i[1] for i in generated_matrix]

data_arr = np.array(data)
class_arr = np.array(classes)

# Separating data values
train_set_x = data_arr[:training_size]  
test_set_x = data_arr[training_size:] 

# Separating class values
train_set_y = class_arr[:training_size]  
test_set_y = class_arr[training_size:] 



# Reshape the training and test examples
train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T


class AIModel:
    def __init__(self, layers = [], epoch=20000, learning_rate=0.5, learning_curve = False, output_cost=True, cost_curve=True, add_momentum=False):
        self.layers = layers
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.output = 0
        self.cost = 0
        self.learning_curve = learning_curve
        self.predictions = []
        self.costs = []
        self.output_cost = output_cost
        self.cost_curve = cost_curve
        self.add_momentum = add_momentum
        self.momentum = []

    def print_cost(self, index):
        if self.output_cost == True:
            print("Cost after {}'th iteration is {}".format(index, self.cost))
            self.cost = 0

    def plot_learn_curve(self):
        plt.plot(self.predictions)
        plt.ylabel('precision')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def plot_cost_curve(self):
        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.build()

    def forward(self):
        input = self.x_train
        for layer in self.layers:
            if self.add_momentum:
                layer.forward(input, self.momentum)
            else:
                layer.forward(input)
            input = layer.output
        self.output = input
        

    
    def build(self):
        
        self.y_train = self.y_train.reshape(len(self.y_train),1)
        for i in range(self.epoch):
            self.forward()
            
            if i % 100 == 0:
                accuracy = len(self.output[self.output == self.y_train]) / len(self.y_train)
                self.predictions.append(accuracy)
                self.costs.append(self.cost)
                self.print_cost(i)

            self.backward()
        if self.learning_curve == True:
            self.plot_learn_curve()
        if self.cost_curve == True:
            self.plot_cost_curve()
    
    def backward(self):
        error = self.y_train - self.output
        dpred_dresult = -error
        self.cost += abs(error.sum())
        
        for layer in self.layers[::-1]:
            layer.backward(self.learning_rate, dpred_dresult)
            dpred_dresult = layer.dpred_dneuron
            self.momentum.append(layer.dpred_dneuron)

    def test(self, x_test, y_test):
        input = x_test
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
        
        pred = input

        return y_test - pred

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)
    
    def set_activation_class(self, Activation):
        self.activation = Activation()

    def forward(self, inputs, add_momentum):
        self.inputs = inputs
        if add_momentum == False:
            self.output = self.activation.forward(np.dot(inputs, self.weights) + self.biases)
        else:
            self.output = self.activation.forward(np.dot(inputs, (self.weights + momentum)) + self.biases)
        self.output[self.output>0.5] = 1
        self.output[self.output<=0.5] = -1


    
    def backward(self, learning_rate, dpred_dresult):
        dresult_dneuron = self.activation.backward(self.output)
        self.dpred_dneuron = dpred_dresult * dresult_dneuron
        self.weights -= learning_rate * np.dot(self.inputs.T, self.dpred_dneuron)


class ActivationSigmoid:
    def forward(self, x):
        return (1 / (np.exp(-x) + 1))
    
    def backward(self, x):
        return (self.forward(x) * (1 - self.forward(x)))

class ActivationReLU:
    def forward(self, x):
        #return 0 if x < 0 else x
        relu = np.vectorize(lambda x: 0 if x < 0 else x)
        return relu(x)
    
    def backward(self, x):
        relu_derivative = np.vectorize(lambda x: 0 if x<0 else 1)
        return relu_derivative(x)


layer = LayerDense(6, 1)
layer.set_activation_class(ActivationSigmoid)

ai_model = AIModel(
    layers=[layer],
    epoch=1000,
    learning_rate=0.0005,
    learning_curve=True,
    add_momentum=True
)

ai_model.train(x_train=train_set_x, y_train=train_set_y)

ai_model.test(test_set_x, test_set_y).sum(keepdims=True, axis = 1)