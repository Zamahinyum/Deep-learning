def train_classifier(net, epochs, lr):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    for epoch in range(epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()

        net.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (net(x).argmax(1) == y).sum().item()
        print(f"Epoch {epoch + 1}/{epochs}  val accuracy {correct / len(val_set):.4f}")
    return net


net = models.resnet50(weights=res_weights)
for p in net.parameters():
    p.requires_grad = False
net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, len(class_names)))

print("Stage 1: feature extraction")
net = train_classifier(net, epochs=15, lr=1e-3)

for p in net.layer4.parameters():
    p.requires_grad = True

print("Stage 2: fine-tuning layer4")
net = train_classifier(net, epochs=15, lr=1e-4)

torch.save(net.state_dict(), "me_vs_not_me.pt")
