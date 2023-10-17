import numpy as np

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
    return thet


# alternative codes 
import numpy as np

num_features = 5
Q = np.zeros(num_features)
learning_rate = 0.01
num_epochs = 1000

# Generate some placeholder data for demonstration
num_samples = 100
X = np.random.rand(num_samples, num_features)
y = np.random.rand(num_samples)

for epoch in range(num_epochs):
    for i in range(num_samples):
        # Calculate the prediction for the current example
        prediction = np.dot(Q, X[i])

        # Update the parameters using the modified gradient descent rule
        gradient = (prediction - y[i]) * X[i]
        Q -= learning_rate * gradient

# Q will be your updated parameters after training
print("Updated parameters (Q):", Q)

