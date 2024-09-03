import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_steps=1000):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Init model parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Gradient descent
        for step in range(self.num_steps):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update model parameters
            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate
            
            if step % 100 == 0:
                cost = 1 / (2 * self.num_steps) * np.sum((y - y_predicted) ** 2)
                print(f"Iteration {step}: Cost = {cost}")
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    
# Generate random dataset
X = np.random.rand(100, 1)
y = 1 + 5 * X + np.random.rand(100, 1) * 0.1

# Create and train model
model = LinearRegression(learning_rate=0.1, num_steps=1000)
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [1]])
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color="b", label="Data")
plt.plot(X_test, y_pred, color="r", label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")