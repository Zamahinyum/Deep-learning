import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

plt.figure(figsize=(10, 3))
for i in range(5):
    img, label = train_set[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[label])
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
    return total_loss / seen, correct / seen


def fit(model, epochs=10, lr=1e-3, weight_decay=0.0, patience=None, loader=None):
    loader = loader or train_loader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch + 1}/{epochs}  loss {tr_loss:.4f}  acc {tr_acc:.4f}  "
              f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}")

        if patience:
            if va_loss < best_loss:
                best_loss, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


model = CNN().to(device)
print(model)

history = fit(model, epochs=10)
test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss())
print(f"Test accuracy: {test_acc:.4f}")
plot_history(history)

model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    probs = torch.softmax(model(images.to(device)), dim=1).cpu()
preds = probs.argmax(1)

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    colour = "green" if preds[i] == labels[i] else "red"
    plt.xlabel(f"{class_names[preds[i]]} {probs[i].max() * 100:.0f}%\n({class_names[labels[i]]})",
               color=colour)
plt.show()
