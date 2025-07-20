import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from kan import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Lennard-Jones potential parameters
epsilon = 1.0
sigma = 1.0

# Generate artificial dataset
def lennard_jones_potential(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Generate r values
r_values = np.linspace(0.8, 3.0, 500)  # Avoid r=0 to prevent singularity

# Compute potential values
potential_values = lennard_jones_potential(r_values, epsilon, sigma)

# Visualize the potential
plt.plot(r_values, potential_values)
plt.xlabel('r')
plt.ylabel('Lennard-Jones Potential')
plt.title('Lennard-Jones Potential')
plt.show()

# Prepare dataset for training
X = r_values.reshape(-1, 1)
y = potential_values.reshape(-1, 1)


model = KAN(width=[1,2,1], grid=5, k=3,)


# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split dataset into train and test sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create DataLoader for train set
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
input_size = 1
output_size = 1
hidden_layers = 2  # Change this value to control the number of hidden layers
neurons_per_layer = 64  # Change this value to control the number of neurons per hidden layer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
train_losses = []
test_losses = []

loss_function = torch.nn.MSELoss()
for epoch in range(num_epochs):
    def closure():
        global loss
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(X_train)),torch.tensor(y_train))
        train_losses.append(loss.detach().numpy())
        loss.backward()
        return loss
    optimizer.step(closure)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    
    # Evaluation
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32))
    test_losses.append(test_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Visualize the loss
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()
plt.show()

# Generate predictions from the model

with torch.no_grad():
    predictions = model(X_tensor).numpy()

# Plot the results
plt.plot(r_values, potential_values, label='Real Lennard-Jones Potential')
plt.plot(r_values, predictions, label='KAN Predictions', linestyle='dashed')
plt.xlabel('r')
plt.ylabel('Lennard-Jones Potential')
plt.title('Lennard-Jones Potential vs KAN Predictions')
plt.legend()
plt.show()

# Plot zoomed-in view of the kink
plt.plot(r_values, potential_values, label='Real Lennard-Jones Potential')
plt.plot(r_values, predictions, label='KAN Predictions', linestyle='dashed')
plt.xlabel('r')
plt.ylabel('Lennard-Jones Potential')
plt.title('Zoomed-in View of Lennard-Jones Potential Minimum')
plt.xlim(0.9, 3)  # Adjust the limits to focus on the kink region
plt.ylim(-1.5, 1.5)  # Adjust the limits to focus on the kink region
plt.legend()
plt.show()
