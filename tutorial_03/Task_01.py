import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    dataset,
    [48000, 12000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class MLP(nn.Module):
    def __init__(self, hidden_layers, activation):
        super().__init__()

        layers = [nn.Flatten()]
        input_size = 784

        for neurons in hidden_layers:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(activation())
            input_size = neurons

        layers.append(nn.Linear(input_size, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct / total)

        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Accuracy: {train_accuracies[-1]:.4f} "
            f"- Validation Accuracy: {val_accuracies[-1]:.4f}"
        )

    return train_losses, val_losses, train_accuracies, val_accuracies


def test_model(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


configs = {
    "Original": ([128, 64], nn.ReLU),
    "More Neurons": ([256, 128], nn.ReLU),
    "More Layers": ([128, 64, 32], nn.ReLU),
    "Tanh": ([128, 64], nn.Tanh),
    "Sigmoid": ([128, 64], nn.Sigmoid)
}

results = {}

for name, (layers, activation) in configs.items():
    print(f"\n{name}")

    model = MLP(layers, activation).to(device)

    train_loss, val_loss, train_acc, val_acc = train_model(model)

    test_accuracy = test_model(model)

    results[name] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_accuracy
    }

    print(f"Test Accuracy: {test_accuracy:.4f}")


plt.figure(figsize=(10, 6))

for name, result in results.items():
    plt.plot(result["val_acc"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy for Different Architectures")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for name, result in results.items():
    plt.plot(result["val_loss"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss for Different Architectures")
plt.legend()
plt.show()
