import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.output_size = output_size
        self.activation = activation
        
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + self.hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []

    def _activation(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'tanh':
            return tanh(x)
        else:
            return sigmoid(x)

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return tanh_derivative(x)
        else:
            return sigmoid_derivative(x)

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                a = z
            else:
                a = self._activation(z)
            self.a_values.append(a)
        
        return a

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        deltas[-1] = self.a_values[-1] - y

        for i in reversed(range(len(self.weights))):
            gradients_w[i] = np.dot(self.a_values[i].T, deltas[i]) / m
            gradients_b[i] = np.sum(deltas[i], axis=0, keepdims=True) / m

            if i > 0:
                deltas[i - 1] = (
                    np.dot(deltas[i], self.weights[i].T)
                    * self._activation_derivative(self.z_values[i - 1])
                )

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, X, y, epochs, learning_rate, log_interval=10):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % log_interval == 0:
                loss = np.mean(np.square(y - output))
                self.loss_history.append(loss)
                self.weight_history.append([w.copy() for w in self.weights])
                self.bias_history.append([b.copy() for b in self.biases])

    def predict(self, X):
        return self.forward(X)

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))