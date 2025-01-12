import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_data = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\linearX.csv', header=None).values
y_data = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\linearY.csv', header=None).values

x_data = (x_data - np.mean(x_data)) / np.std(x_data)
x_data = np.c_[np.ones(x_data.shape[0]), x_data]

def gradient_descent(x, y, learning_rate, iterations):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    y = y.reshape(-1)

    for i in range(iterations):
        h = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
        cost_history.append(cost)
        gradient = (1 / m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient
        if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < 1e-6:
            break

    return theta, cost_history

learning_rate = 0.05
iterations = 50
theta, cost_history = gradient_descent(x_data, y_data, learning_rate, iterations)

plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost Function (J)")
plt.title(f"Cost Function vs Iteration (lr={learning_rate})")
plt.show()

plt.scatter(x_data[:, 1], y_data, color='blue', label="Data points")
x_line = np.linspace(min(x_data[:, 1]), max(x_data[:, 1]), 100)
y_line = theta[0] + theta[1] * x_line
plt.plot(x_line, y_line, color='red', label="Fitted line")
plt.xlabel("Predictor")
plt.ylabel("Response")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

learning_rates = [0.005, 0.05, 0.1]
for lr in learning_rates:
    theta, cost_history = gradient_descent(x_data, y_data, lr, 50)
    plt.plot(range(len(cost_history)), cost_history, label=f"lr={lr}")
plt.xlabel("Iteration")
plt.ylabel("Cost Function (J)")
plt.title("Cost Function vs Iteration for different Learning Rates")
plt.legend()
plt.show()
