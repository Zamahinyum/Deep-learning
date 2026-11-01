train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tfm)
val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tfm)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=8, num_workers=0)
class_names = train_set.classes
print(class_names)
