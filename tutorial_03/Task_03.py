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
    def __init__(self, dropout=0):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


def train_model(epochs=20, dropout=0, weight_decay=0, patience=3):
    model = MLP(dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=weight_decay
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

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

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Accuracy: {train_accuracies[-1]:.4f} "
            f"- Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


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


experiments = {
    "Original": {
        "dropout": 0,
        "weight_decay": 0
    },
    "Dropout": {
        "dropout": 0.3,
        "weight_decay": 0
    },
    "L2 Regularization": {
        "dropout": 0,
        "weight_decay": 0.0001
    }
}

results = {}

for name, settings in experiments.items():
    print(f"\n{name}")

    model, train_loss, val_loss, train_acc, val_acc = train_model(
        epochs=20,
        dropout=settings["dropout"],
        weight_decay=settings["weight_decay"]
    )

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
    plt.plot(result["train_acc"], label=f"{name} - Train")
    plt.plot(result["val_acc"], linestyle="--", label=f"{name} - Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for name, result in results.items():
    plt.plot(result["train_loss"], label=f"{name} - Train")
    plt.plot(result["val_loss"], linestyle="--", label=f"{name} - Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
