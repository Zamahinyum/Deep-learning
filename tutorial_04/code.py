import os
import matplotlib.pyplot as plt
import torch.nn as nn
from google.colab import files
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image

save_folder = "/content/augmented_images"
os.makedirs(save_folder, exist_ok=True)

datagen = transforms.Compose([
    transforms.RandomAffine(degrees=40, shear=20, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.5, 1.5)),
])

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
)
print(model)

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img = Image.open(image_path).convert("RGB")

x = to_tensor(img)

num_images = 40
fig, axes = plt.subplots(8, 5, figsize=(15, 24))

for i, ax in enumerate(axes.flat):
    augmented = datagen(x)
    save_image(augmented, os.path.join(save_folder, f"image_{i}.jpeg"))
    ax.imshow(augmented.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(f"image_{i}")
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"{len(os.listdir(save_folder))} augmented images are saved in the folder: {os.path.abspath(save_folder)}")

!zip -rq augmented_images.zip /content/augmented_images
files.download("augmented_images.zip")
