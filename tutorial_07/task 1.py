aug_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
aug_train = datasets.CIFAR10("./data", train=True, transform=aug_tfm)
aug_loader = DataLoader(Subset(aug_train, train_idx.indices), batch_size=64, shuffle=True, num_workers=2)

improved = build_resnet50_feature_extractor()
for block in (improved.layer3, improved.layer4):
    for p in block.parameters():
        p.requires_grad = True
improved = improved.to(device)

optimizer = optim.AdamW([
    {"params": improved.layer3.parameters(), "lr": 1e-5},
    {"params": improved.layer4.parameters(), "lr": 5e-5},
    {"params": improved.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-4)

epochs = 12
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

improved, improved_hist = fit(improved, optimizer, aug_loader, val_loader,
                              epochs=epochs, patience=4, scheduler=scheduler, criterion=criterion)
results["ResNet50 improved fine-tuning"] = evaluate(improved, test_loader)[1]
print(f"Improved fine-tuning test accuracy: {results['ResNet50 improved fine-tuning']:.4f}")
