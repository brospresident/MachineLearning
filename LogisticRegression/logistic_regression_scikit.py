import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Prepare the data
X = np.array([1, 2, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 1, 1, 1])

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 2: Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Step 3: Make predictions
y_pred = model.predict(X)

# Step 4: Evaluate the model
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 5: Print model parameters
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient: {model.coef_[0][0]:.4f}")

# Step 6: Visualize the results
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(X, y, color='b', label='Training Data')

# Plot the decision boundary
X_plot = np.linspace(0, 7, 100).reshape(-1, 1)
y_plot = model.predict_proba(X_plot)[:, 1]
plt.plot(X_plot, y_plot, color='r', label='Decision Boundary')

plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Exam Pass Prediction (sklearn)')
plt.legend()

# Step 7: Make a prediction for a student who studied for 3 hours
hours_studied = 3
prob_pass = model.predict_proba(np.array([[hours_studied]]))
print(f"Probability of passing after studying for {hours_studied} hours: {prob_pass[0][1]:.4f}")

plt.show()
