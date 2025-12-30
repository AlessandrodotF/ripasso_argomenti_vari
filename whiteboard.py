from torchvision.models import resnet50, ResNet50_Weights

N_CLASSES = 10
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
print(model)
