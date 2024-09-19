import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Generate data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.float32).reshape(-1, 1)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create the network
net = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Train the network
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the network
test_input = torch.FloatTensor([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]])
with torch.no_grad():
    predictions = net(test_input)
print("\nPredictions:")
print(predictions.numpy())