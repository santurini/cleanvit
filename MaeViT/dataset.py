import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
from PIL import Image
from helpers import pair
import math

class Data(Dataset):
  def __init__(self, path, img_size=128, split='train', size=0.8):
    self.path = list(sorted(Path(path).rglob('*')))
    split_point = int(len(self.path)*size)
    self.size = pair(img_size)
    if split=='train': self.path = self.path[:split_point]
    else: self.path = self.path[split_point:]
    
  def __len__(self):
    return len(self.path)
    
  def __getitem__(self, idx): 
    x = self.load_img(self.path[idx])
    return self.crop(x)
  
  def load_img(self, path):
    return to_tensor(Image.open(path))

  def crop(self, img):
    height = img.shape[1]
    width = img.shape[2]
    y_offset = int(math.ceil((height - self.size[0]) / 2))
    x_offset = int(math.ceil((width - self.size[1]) / 2))
    return img[:, y_offset : y_offset + self.size[0], x_offset : x_offset + self.size[1]]

