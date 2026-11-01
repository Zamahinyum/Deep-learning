import os
import random
import shutil
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = r"D:\1stsemester\Machine Learning\assignment 2\dataset\me\8.jpg"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def top5(model, weights, image):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0]
    scores, idx = probs.topk(5)
    labels = weights.meta["categories"]
    return [(labels[i], s.item()) for i, s in zip(idx, scores)]


def show_top5(title, model, weights, image):
    print(title)
    for name, score in top5(model, weights, image):
        print(f"  {name}: {score * 100:.2f}%")


image = load_image(image_path)

vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
res_weights = models.ResNet50_Weights.IMAGENET1K_V1

vgg16 = models.vgg16(weights=vgg_weights).to(device)
resnet50 = models.resnet50(weights=res_weights).to(device)

show_top5("VGG16", vgg16, vgg_weights, image)
show_top5("ResNet50", resnet50, res_weights, image)
