import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.network(x)


learning_rates = [0.0001, 0.001, 0.01, 0.1]

results = {}
loss_curves = {}

max_epochs = 1000
tolerance = 1e-6
patience = 20


for lr in learning_rates:

    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    losses = []
    best_loss = float("inf")
    patience_count = 0

    for epoch in range(max_epochs):

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        if best_loss - current_loss > tolerance:
            best_loss = current_loss
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

        accuracy = (predictions == y_test).float().mean().item()

    results[lr] = {
        "accuracy": accuracy,
        "epochs": len(losses),
        "loss": losses[-1]
    }

    loss_curves[lr] = losses


print("\nTask 2 Results")
print("-" * 50)

for lr, result in results.items():

    print(f"\nLearning Rate: {lr}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Epochs: {result['epochs']}")
    print(f"Final Loss: {result['loss']:.6f}")


plt.figure(figsize=(10, 6))

for lr, losses in loss_curves.items():
    plt.plot(
        losses,
        label=f"Learning Rate = {lr}"
    )

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Effect of Learning Rate on MLP Convergence")
plt.legend()
plt.grid()

plt.show()
