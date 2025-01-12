import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_data = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\linearX.csv', header=None).values
y_data = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\linearY.csv', header=None).values

x_data = (x_data - np.mean(x_data)) / np.std(x_data)
x_data = np.c_[np.ones(x_data.shape[0]), x_data]

def gradient_descent(x, y, learning_rate, max_iterations):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    y = y.reshape(-1)
    for _ in range(max_iterations):
        h = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
        cost_history.append(cost)
        gradient = (1 / m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient
    return theta, cost_history

def stochastic_gradient_descent(x, y, learning_rate, max_iterations):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    y = y.reshape(-1)
    for i in range(max_iterations):
        for j in range(m):
            index = np.random.randint(m)
            xi = x[index:index + 1]  # Ensure xi is a row vector
            yi = y[index]           # yi is a scalar
            h = np.dot(xi, theta)
            gradient = np.dot(xi.T, (h - yi))  # Gradient is a column vector
            theta -= learning_rate * gradient.flatten()  # Flatten to match theta's shape
        h_full = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((h_full - y) ** 2)
        cost_history.append(cost)
    return theta, cost_history


def mini_batch_gradient_descent(x, y, learning_rate, max_iterations, batch_size):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    y = y.reshape(-1)
    for _ in range(max_iterations):
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            xi = x_shuffled[i:i + batch_size]  # Select batch of predictors
            yi = y_shuffled[i:i + batch_size]  # Select batch of responses
            h = np.dot(xi, theta)
            gradient = np.dot(xi.T, (h - yi)) / len(yi)  # Average gradient over the batch
            theta -= learning_rate * gradient
        h_full = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((h_full - y) ** 2)
        cost_history.append(cost)
    return theta, cost_history


learning_rate = 0.5
max_iterations = 50
batch_size = 16

theta_bgd, cost_bgd = gradient_descent(x_data, y_data, learning_rate, max_iterations)
theta_sgd, cost_sgd = stochastic_gradient_descent(x_data, y_data, learning_rate, max_iterations)
theta_mbgd, cost_mbgd = mini_batch_gradient_descent(x_data, y_data, learning_rate, max_iterations, batch_size)

plt.plot(range(len(cost_bgd)), cost_bgd, label="Batch Gradient Descent", color='blue')
plt.plot(range(len(cost_sgd)), cost_sgd, label="Stochastic Gradient Descent", color='red')
plt.plot(range(len(cost_mbgd)), cost_mbgd, label="Mini-Batch Gradient Descent", color='green')
plt.xlabel("Iterations")
plt.ylabel("Cost Function (J)")
plt.title("Cost Function vs Iterations for Different Gradient Descent Methods")
plt.legend()
plt.show()
