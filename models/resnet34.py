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

class Res(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Res, self).__init__()
        res = models.resnet34(pretrained=True)
        res.fc = nn.Linear(512,10)
        self.go = res

        
    def forward(self, x):
        x = self.go(x)
        return x

