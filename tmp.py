import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def ReLU(x):
    return np.maximum(0, x)

# tanh function
def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-5, 5, 100)

# Calculate sigmoid values
sigmoid_values = sigmoid(x)

# Calculate ReLU values
ReLU_values = ReLU(x)

# Calculate tanh values
tanh_values = tanh(x)

# Set up the subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot sigmoid function
ax[0].plot(x, sigmoid_values)
ax[0].set_title('Sigmoid Function')

# Plot ReLU function
ax[1].plot(x, ReLU_values)
ax[1].set_title('ReLU Function')

# Plot tanh function
ax[2].plot(x, tanh_values)
ax[2].set_title('tanh Function')

# Show the plot
plt.show()
