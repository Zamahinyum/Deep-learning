import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_tfm)
val_base = datasets.CIFAR10("./data", train=True, transform=eval_tfm)
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tfm)

gen = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(range(len(full_train)), [45000, 5000], generator=gen)

train_loader = DataLoader(Subset(full_train, train_idx.indices), batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(Subset(val_base, val_idx.indices), batch_size=64, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, num_workers=2)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def evaluate(model, loader, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
    return loss_sum / seen, correct / seen


def fit(model, optimizer, train_loader, val_loader, epochs=5, patience=3,
        scheduler=None, criterion=None):
    model = model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    best_loss, best_state, wait = float("inf"), None, 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        val_loss, val_acc = evaluate(model, val_loader)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(loss_sum / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}  loss {loss_sum / seen:.4f}  "
              f"acc {correct / seen:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model, history


def build_vgg16_feature_extractor():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    freeze(model)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)
    )
    return model


def build_resnet50_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    freeze(model)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)
    )
    return model


def build_resnet50_finetune():
    model = build_resnet50_feature_extractor()
    for p in model.layer4[-1].parameters():
        p.requires_grad = True
    return model


results = {}

vgg_fe = build_vgg16_feature_extractor().to(device)
vgg_fe, vgg_fe_hist = fit(vgg_fe, optim.Adam(trainable(vgg_fe), lr=1e-3), train_loader, val_loader)
results["VGG16 feature extraction"] = evaluate(vgg_fe, test_loader)[1]

res_ft = build_resnet50_finetune().to(device)
res_ft, res_ft_hist = fit(res_ft, optim.Adam(trainable(res_ft), lr=1e-4), train_loader, val_loader)
results["ResNet50 fine-tuning"] = evaluate(res_ft, test_loader)[1]

for name, acc in results.items():
    print(f"{name:30s} test accuracy {acc:.4f}")

plt.plot(vgg_fe_hist["val_acc"], label="VGG16 feature extraction")
plt.plot(res_ft_hist["val_acc"], label="ResNet50 fine-tuning")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.legend()
plt.show()
