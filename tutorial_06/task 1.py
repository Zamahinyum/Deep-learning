alex_weights = models.AlexNet_Weights.IMAGENET1K_V1
res101_weights = models.ResNet101_Weights.IMAGENET1K_V1
mobile_weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

others = {
    "AlexNet": (models.alexnet(weights=alex_weights), alex_weights),
    "ResNet101": (models.resnet101(weights=res_weights if False else res101_weights), res101_weights),
    "MobileNetV2": (models.mobilenet_v2(weights=mobile_weights), mobile_weights),
}

for title, (net, w) in others.items():
    show_top5(title, net.to(device), w, image)
