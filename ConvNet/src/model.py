import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10ConvNet10(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()
    # Block 1
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(16)
    self.drop1 = nn.Dropout(0.2)
    
    # Block 2
    self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # downsample
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)
    self.drop2 = nn.Dropout(0.2)

    # Block 3
    self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # downsample
    self.bn5 = nn.BatchNorm2d(64)
    self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(64)
    self.drop3 = nn.Dropout(0.2)

    # Block 4
    self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # downsample
    self.bn7 = nn.BatchNorm2d(128)
    self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(128)
    self.drop4 = nn.Dropout(0.2)

    # Final conv block for embedding
    self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn9 = nn.BatchNorm2d(128)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.drop = nn.Dropout(0.5)
    self.fc = nn.Linear(128, num_classes)

  def forward(self, x):
    # Block 1
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.drop1(x)

    # Block 2
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = self.drop2(x)

    # Block 3
    x = F.relu(self.bn5(self.conv5(x)))
    x = F.relu(self.bn6(self.conv6(x)))
    x = self.drop3(x)

    # Block 4
    x = F.relu(self.bn7(self.conv7(x)))
    x = F.relu(self.bn8(self.conv8(x)))
    x = F.relu(self.bn9(self.conv9(x)))
    x = self.drop4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.drop(x)
    x = self.fc(x)

    return x