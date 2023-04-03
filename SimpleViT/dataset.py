import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchvision.transforms import Resize
from helpers import pair
import numpy as np

class Compressed(Dataset):
  def __init__(self, path, shape=(256, 256), split='train', size=0.8, seed=123):
    self.hq_path = np.array(sorted(x for x in (Path(path) / 'HR').rglob('*') if x.is_file()))
    self.lq_path = np.array(sorted(x for x in (Path(path) / 'LR').rglob('*') if x.is_file()))
    
    random.seed(seed)
    permutation = np.array(range(len(self.hq_path)))
    np.random.shuffle(permutation)
    self.lq_path = self.lq_path[permutation]
    self.hq_path = self.hq_path[permutation]
    
    split_point = int(len(self.hq_path)*size)
    if split=='train':
        self.lq_path = self.lq_path[:split_point]
        self.hq_path = self.hq_path[:split_point]
    else:
        self.lq_path = self.lq_path[split_point:]
        self.hq_path = self.hq_path[split_point:]
    
    self.resize = Resize(pair(shape))
    
  def __len__(self):
    return len(self.lq_path)
    
  def __getitem__(self, idx):  
    assert self.lq_path[idx].name == self.hq_path[idx].name
    lq_img = self.load_img(self.lq_path[idx])
    hq_img = self.resize(self.load_img(self.hq_path[idx]))
    return lq_img, hq_img
  
  def load_img(self, path):
    return to_tensor(Image.open(path))
