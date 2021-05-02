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

class Vgg16(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Vgg16, self).__init__()
        vgg = models.vgg16(pretrained=True)
       
        vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.Linear(4096,1024),
            nn.Linear(1024,10)
        )
        self.vgg = vgg

        
    def forward(self, x):
        x = self.vgg(x)

        return x

