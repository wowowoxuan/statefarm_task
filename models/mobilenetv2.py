import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision

class Mobilenet_v2(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Mobilenet_v2, self).__init__()
        self.mobnet = models.mobilenet_v2(pretrained=True)
        self.fc1 = nn.Linear(1000, 10)

        
    def forward(self, x):
        x = self.mobnet(x)
        x = self.fc1(x)
        return x

