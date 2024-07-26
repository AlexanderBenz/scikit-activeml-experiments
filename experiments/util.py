import torch
import torchvision.transforms as transforms

def get_transformer_by_name(name):
    transformer = None
    if name == "dinov2_vitb14":
        transformer = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
    elif name == "cifar10":
        transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    return transformer