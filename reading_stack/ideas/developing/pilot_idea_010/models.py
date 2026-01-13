"""
Model definitions for IDEA-010 pilot experiment.

SimpleCNN: Fast iteration model (~100K params)
ResNet18CIFAR: ResNet-18 adapted for CIFAR-10 (32x32 images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10. ~100K parameters for fast iteration.

    Architecture:
        Conv(3→32, 3x3) → ReLU → MaxPool(2x2)
        Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
        Conv(64→64, 3x3) → ReLU → MaxPool(2x2)
        FC(64*4*4→256) → ReLU → Dropout → FC(256→10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).

    Modifications from ImageNet ResNet-18:
        - First conv: 3x3, stride 1, padding 1 (instead of 7x7, stride 2)
        - No initial max pooling
        - Smaller input size throughout
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Load pretrained=False ResNet-18
        self.resnet = resnet18(weights=None, num_classes=num_classes)

        # Modify first conv for CIFAR-10 (32x32 instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove the max pooling layer (identity)
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to get model by name.

    Args:
        model_name: 'simple_cnn' or 'resnet18'
        num_classes: Number of output classes

    Returns:
        Initialized model
    """
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        return ResNet18CIFAR(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation and parameter counts
    for name in ['simple_cnn', 'resnet18']:
        model = get_model(name)
        params = count_parameters(model)

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)

        print(f"{name}: {params:,} parameters, output shape: {y.shape}")
