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

class weightcomb(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(weightcomb, self).__init__()
        self.fc = nn.Linear(20,10)


        
    def forward(self, x):
        x = self.fc(x)
        return x

