import torch
import torchvision.transforms as transforms

def get_transformer_by_name(name):
    transformer = None
    if "dinov2" in name:
        transformer = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
        
    return transformer