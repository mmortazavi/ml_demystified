import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gradient Descent Demo")

# Generate some sample data
np.random.seed(42)
x = np.random.rand(100)
y = 2*x + 0.5*np.random.randn(100)

# Define the linear regression model
def linear_regression(x, w, b):
    return w*x + b

# Define the mean squared error loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Define the gradient of the loss function with respect to the parameters
def gradient(x, y_pred, y_true):
    dw = np.mean((y_pred - y_true)*x)
    db = np.mean(y_pred - y_true)
    return dw, db

# Define the gradient descent algorithm
def gradient_descent(x, y, w_init, b_init, learning_rate, num_iterations):
    losses = []
    weights = [w_init]
    biases = [b_init]
    w = w_init
    b = b_init
    for i in range(num_iterations):
        y_pred = linear_regression(x, w, b)
        loss = mse_loss(y_pred, y)
        dw, db = gradient(x, y_pred, y)
        w -= learning_rate*dw
        b -= learning_rate*db
        weights.append(w)
        biases.append(b)
        losses.append(loss)
    return weights, biases, losses

# Define the hyperparameters
learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
num_iterations = st.slider("Number of Iterations", 100, 1000, 500, 10)

# Run the gradient descent algorithm
w_init = 0.0
b_init = 0.0
weights, biases, losses = gradient_descent(x, y, w_init, b_init, learning_rate, num_iterations)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(losses)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y)
for i in range(len(weights)):
    ax.plot(x, linear_regression(x, weights[i], biases[i]), color="red", alpha=i/len(weights))
ax.set_xlabel("X")
ax.set_ylabel("Y")
st.pyplot(fig)
