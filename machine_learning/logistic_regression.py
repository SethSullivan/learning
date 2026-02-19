import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
import data_visualization as dv

dv.set_plot_style("cashaback_dark.mplstyle")

# %% Generate synthetic binary classification data
# Here we have X as 569 samples with 30 features for breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# %% Set up logistic function, loss, and grads


def sigmoid(X: np.ndarray, W: np.ndarray, b: float):
    """Logistic function

    Args:
        X: n sample x m feature array
        W: m feature x n sample weights array
        b: bias term

    Returns:
        float
    """
    linear = np.dot(X, W) + b
    return 1 / (1 + np.e ** (-linear))


def predict(X: np.ndarray, W: np.ndarray, b: float, threshold=0.5):
    """Predict binary class labels based on logistic function output

    Args:
        X: n sample x m feature array
        W: m feature x n sample weights array
        b: bias term
        threshold: decision threshold for classification

    Returns:
        np.ndarray: Predicted class labels (0 or 1)
    """
    probabilities = sigmoid(X, W, b)
    return (probabilities >= threshold).astype(int)


def logloss(y: np.ndarray, predictions: np.ndarray, epsilon=1e-6):
    """Compute the log loss (cross-entropy loss) for binary classification"""
    # Makes sense since np.log(1) = 0
    # y is 0 or 1.
    # So if y = 1, then we use the first term. If prediction is 1, that's zero loss bc we got it!
    # print(predictions)
    return -np.mean(
        y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon)
    )


def grad_logloss(y, X, predictions, num_features):
    """Compute the gradient of log loss with respect to weights W

    I decided to do this inline with dot product, but this works too.
    This calculates the gradient of the loss function with respect to each parameter (feature)
    """
    grads = np.empty(num_features)
    for i in range(num_features):
        grads[i] = np.sum((predictions - y) * X[:, i])
    return grads


def fit(y, X, W, b, lr=0.000001):
    output = sigmoid(X, W, b)
    loss = logloss(y, output)
    gradient = np.dot(X.T, (output - y)) / y.shape[0]

    next_weights = W - lr * gradient

    return output, loss, gradient, next_weights


# %%

num_features = X.shape[1]
num_datapoints = X.shape[0]
b = 0

num_iterations = 10000
model_output = np.empty((num_iterations, num_datapoints))
losses = np.empty(num_iterations)
gradients = np.empty((num_iterations, num_features))
weights = np.zeros((num_iterations + 1, num_features))
accuracies = np.empty(num_iterations)
for i in range(num_iterations):
    model_output[i], losses[i], gradients[i], weights[i + 1] = fit(y, X, weights[i], b)

    predicted_labels = predict(np.array(X), weights[i], b)
    accuracies[i] = np.sum(predicted_labels == y) / num_datapoints

print(f"Final Accuracy: {accuracies[-1]:.4f}")
# %% Plotting the loss curve

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Log Loss Curve")
plt.show()

plt.plot(accuracies)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.show()

# %% Plot predicted probabilities vs true labels
plt.scatter(model_output[-1], y, alpha=0.5)
plt.xlabel("Predicted Probability")
plt.ylabel("True Label")
plt.title("Predicted Probabilities vs True Labels")
plt.show()

# %% Feature importance based on final weights
# We can get an idea of what matters for classifying breast cancer.
# Absolute value gets you the importance, but its likely also worth looking at the sign
final_weights = weights[-1]
feature_importance = np.abs(final_weights)
colors = ["blue" if w > 0 else "red" for w in final_weights]
fig, ax = plt.subplots()
ax.bar(np.arange(num_features), feature_importance, color=colors)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Absolute Weight")
ax.set_title("Feature Importance Based on Final Weights")
dv.legend(ax, ["Positive Weight", "Negative Weight"], colors=["blue", "red"])
plt.show()
