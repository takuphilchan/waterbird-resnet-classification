import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class WaterBirdModel(nn.Module):
    """ResNet18-based model for WaterBird classification"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18 with new weights parameter
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)