vgg_ft = build_vgg16_feature_extractor()
for p in vgg_ft.features[24:].parameters():
    p.requires_grad = True
vgg_ft = vgg_ft.to(device)

vgg_ft, vgg_ft_hist = fit(vgg_ft, optim.Adam(trainable(vgg_ft), lr=1e-4), train_loader, val_loader)
results["VGG16 fine-tuning"] = evaluate(vgg_ft, test_loader)[1]

print("\nFinal comparison")
for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{name:32s} {acc:.4f}")

plt.figure(figsize=(10, 4))
plt.bar(range(len(results)), list(results.values()))
plt.xticks(range(len(results)), [n.replace(" ", "\n") for n in results], fontsize=7)
plt.ylabel("Test accuracy")
plt.tight_layout()
plt.show()
