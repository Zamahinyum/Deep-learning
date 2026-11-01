def predict(path):
    x = val_tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    for name, p in zip(class_names, probs):
        print(f"{name}: {p.item() * 100:.1f}%")


predict(image_path)
