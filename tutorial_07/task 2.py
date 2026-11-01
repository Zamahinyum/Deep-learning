class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


res_fe = build_resnet50_feature_extractor().to(device)
res_fe, res_fe_hist = fit(res_fe, optim.Adam(trainable(res_fe), lr=1e-3), train_loader, val_loader)
results["ResNet50 feature extraction"] = evaluate(res_fe, test_loader)[1]

custom = SmallCNN().to(device)
custom, custom_hist = fit(custom, optim.Adam(custom.parameters(), lr=1e-3),
                          train_loader, val_loader, epochs=10)
results["Custom CNN"] = evaluate(custom, test_loader)[1]

for name in ["Custom CNN", "VGG16 feature extraction", "ResNet50 feature extraction"]:
    print(f"{name:30s} {results[name]:.4f}")
