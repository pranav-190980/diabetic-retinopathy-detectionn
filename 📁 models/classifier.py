import torch.nn as nn
from torchvision import models

def get_model(model_name="efficientnet"):
    """
    Returns a pretrained model for classification
    Classes = 5 (DR stages)
    """

    if model_name == "resnet":
        model = models.resnet50(pretrained=True)

        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, 5)

    else:
        model = models.efficientnet_b0(pretrained=True)

        # Replace classifier layer
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    return model
