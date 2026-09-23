import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)

print("\nDataset shape:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 scaled training samples:")
print(X_train_scaled[:5])

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)

# Train the model
mlp.fit(X_train_scaled, y_train)

print("\nMLP model trained successfully!")
y_pred = mlp.predict(X_test_scaled)

print("\nPredicted values:")
print(y_pred)

print("\nActual values:")
print(y_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


print("\n========== MLP MODEL INFORMATION ==========")

print("Number of layers:",
      mlp.n_layers_)

print("Number of outputs:",
      mlp.n_outputs_)

print("Hidden layer sizes:",
      mlp.hidden_layer_sizes)

print("Activation function:",
      mlp.activation)

print("Number of iterations:",
      mlp.n_iter_)

print("Final loss:",
      mlp.loss_)

print("Training score:",
      mlp.score(X_train_scaled, y_train))

print("Testing score:",
      mlp.score(X_test_scaled, y_test))

plt.figure(figsize=(8, 5))

plt.plot(
    mlp.loss_curve_,
    label='Training Loss'
)

plt.xlabel("Epochs / Iterations")
plt.ylabel("Loss")
plt.title("MLP Learning Curve")

plt.legend()
plt.grid()

plt.show()
