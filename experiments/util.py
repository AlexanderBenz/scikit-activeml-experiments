import torch
import torchvision.transforms as transforms
"""
If a new torch backbone is needed please add them to this file.
"""
def get_transformer_by_name(name):
    """Function to return a transformer based on the backbone model"""
    transformer = None
    if "dinov2" in name:
        transformer = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
        
    return transformer