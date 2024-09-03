import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate random dataset
X = np.random.rand(100, 1)
y = 1 + 5 * X + np.random.rand(100, 1) * 0.1

# Create and train model
model = LinearRegression()
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

