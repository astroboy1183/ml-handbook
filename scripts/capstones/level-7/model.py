"""Level 7 capstone: models.

SmallResNet is a ResNet-style CNN sized for a CPU: a 3x3 stem, then three
residual stages at 28x28, 14x14, and 7x7 resolution, global average pooling,
and one linear layer. LinearBaseline is softmax regression on raw pixels.
"""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 conv-BN layers plus a shortcut; a 1x1 conv shortcut when the shape changes."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, width=16, num_classes=10, dropout=0.1):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.stages = nn.Sequential(
            ResidualBlock(w, w, 1),          # 28x28
            ResidualBlock(w, 2 * w, 2),      # 14x14
            ResidualBlock(2 * w, 4 * w, 2),  # 7x7
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(4 * w, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class LinearBaseline(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, num_classes))

    def forward(self, x):
        return self.net(x)


def build_model(name="resnet", width=16):
    if name == "linear":
        return LinearBaseline()
    return SmallResNet(width=width)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
