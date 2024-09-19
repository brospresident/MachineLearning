import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.dZ2 = output - y
        self.dW2 = np.dot(self.a1.T, self.dZ2)
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True)
        self.dZ1 = np.dot(self.dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        self.dW1 = np.dot(X.T, self.dZ1)
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True)
    
    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            self.W1 -= learning_rate * self.dW1
            self.b1 -= learning_rate * self.db1
            self.W2 -= learning_rate * self.dW2
            self.b2 -= learning_rate * self.db2
    
    def predict(self, X):
        return self.forward(X)

# Example usage
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int).reshape(-1, 1)

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, learning_rate=0.1, epochs=10000)

# Test the network
test_input = np.array([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]])
predictions = nn.predict(test_input)
print("Predictions:")
print(predictions)