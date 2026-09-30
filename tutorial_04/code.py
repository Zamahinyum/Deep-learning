import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image
import requests
from io import BytesIO

# Define a save folder within the Colab environment
save_folder = "/content/augmented_images"
os.makedirs(save_folder, exist_ok=True)

datagen = transforms.Compose([
    transforms.RandomAffine(degrees=40, shear=0.2, scale=(0.8, 1.2)),
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

# Public URL for an image
image_url = "https://as1.ftcdn.net/jpg/03/74/30/90/1000_F_374309012_hlGrwhklbM7vpxlhZJwhVZBS9dWj5g0d.jpg"

# Download the image from the URL
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert("RGB")

x = to_tensor(img)
x = x.unsqueeze(0)

i = 0
while True:
    batch = datagen(x)
    save_image(batch, os.path.join(save_folder, f"image_{i}.jpeg"))
    # Convert tensor to PIL Image for displaying with matplotlib
    augmented_image = (batch[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    plt.figure()
    plt.imshow(augmented_image)
    plt.axis('off')
    plt.title(f"Augmented Image {i}") # Add a title to each plot
    plt.show()
    i += 1
    if i > 20:
        break

print(f"Augmented images are saved in the folder: {os.path.abspath(save_folder)}")
