import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)
X = np.hstack((np.ones((X.shape[0], 1)), X))

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m) * (-y.T @ np.log(h + epsilon) - (1-y).T @ np.log(1-h + epsilon))
    return cost.item()

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1/m) * X.T @ (h - y)
        theta -= alpha * gradient
        J_history.append(cost(X, y, theta))
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {J_history[-1]}")
            print(f"Theta shape: {theta.shape}, Gradient shape: {gradient.shape}")
    
    return theta, J_history

theta = np.zeros((X.shape[1], 1))
alpha = 0.1
num_iters = 1000

print(X.shape)
print(y.shape)
print(theta.shape)

theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

def predict(X, theta):
    return sigmoid(X @ theta)

plt.figure(figsize=(10, 8))

plt.scatter(X[:, 1], y, color='b', label='Training Data')

x_values = np.array([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
y_values = -(theta[0] + theta[1] * x_values) / theta[1]
plt.plot(x_values, sigmoid(y_values), color='r', label='Decision Boundary')

plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Exam Pass Prediction')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence of Cost Function')

print(f"Final Parameters: theta0 = {theta[0][0]:.4f}, theta1 = {theta[1][0]:.4f}")

hours_studied = 3
prob_pass = predict(np.array([1, hours_studied]), theta)
print(f"Probability of passing after studying for {hours_studied} hours: {prob_pass[0]:.4f}")

plt.show()
    
