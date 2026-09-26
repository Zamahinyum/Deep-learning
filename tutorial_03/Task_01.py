import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


torch.manual_seed(42)

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_data, val_data = random_split(
    train_data,
    [48000, 12000]
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=32
)

test_loader = DataLoader(
    test_data,
    batch_size=32
)


class MLP(nn.Module):

    def __init__(self, hidden_layers, activation):
        super().__init__()

        layers = [
            nn.Flatten()
        ]

        input_size = 784

        for neurons in hidden_layers:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(activation())
            input_size = neurons

        layers.append(nn.Linear(input_size, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


configurations = {
    "Original": ((128, 64), nn.ReLU),
    "More Neurons": ((256, 128), nn.ReLU),
    "More Layers": ((128, 64, 32), nn.ReLU),
    "Tanh": ((128, 64), nn.Tanh),
    "Sigmoid": ((128, 64), nn.Sigmoid)
}


def train_model(model, epochs=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies
    )


results = {}

for name, (layers, activation) in configurations.items():

    print(f"Training: {name}")

    model = MLP(layers, activation)

    history = train_model(model)

    results[name] = history

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}\n")


plt.figure(figsize=(10, 6))

for name, history in results.items():
    plt.plot(
        history[1],
        label=name
    )

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss - Different Architectures")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))

for name, history in results.items():
    plt.plot(
        history[3],
        label=name
    )

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy - Different Architectures")
plt.legend()
plt.grid()
plt.show()
