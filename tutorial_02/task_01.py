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

    def __init__(self, hidden_layers):
        super().__init__()

        layers = []
        input_size = 4

        for neurons in hidden_layers:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(nn.ReLU())
            input_size = neurons

        layers.append(nn.Linear(input_size, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


configurations = {
    "10 neurons": (10,),
    "20 neurons": (20,),
    "10, 10": (10, 10),
    "20, 20": (20, 20),
    "10, 10, 10": (10, 10, 10)
}

results = {}
loss_curves = {}

for name, layers in configurations.items():

    model = MLP(layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(1000):

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

        accuracy = (predictions == y_test).float().mean().item()

    results[name] = {
        "accuracy": accuracy,
        "epochs": len(losses),
        "loss": losses[-1]
    }

    loss_curves[name] = losses


print("\nTask 1 Results")
print("-" * 50)

for name, result in results.items():
    print(f"\nConfiguration: {name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Epochs: {result['epochs']}")
    print(f"Final Loss: {result['loss']:.6f}")


best = max(results, key=lambda x: results[x]["accuracy"])

print("\nBest Configuration")
print("-" * 50)
print(f"Configuration: {best}")
print(f"Accuracy: {results[best]['accuracy']:.4f}")


plt.figure(figsize=(10, 6))

for name, losses in loss_curves.items():
    plt.plot(losses, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Effect of Hidden Layers and Neurons")
plt.legend()
plt.grid()

plt.show()
