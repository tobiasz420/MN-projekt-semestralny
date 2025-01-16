import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn, optim
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset): 

    def __init__(self, image_dir, label_dir=None, transform=None): 

        self.image_dir = image_dir 

        self.label_dir = label_dir 

        self.images = os.listdir(image_dir) 

        self.transform = transform 

 

    def __len__(self): 

        return len(self.images) 

 

    def __getitem__(self, index): 

        image_path = os.path.join(self.image_dir, self.images[index]) 

        image = Image.open(image_path).convert("RGB") 

 

        label = None 

        if self.label_dir: 

            label_path = os.path.join(self.label_dir, self.images[index].replace('.png', '.txt')) 

            if os.path.exists(label_path): 

                with open(label_path, 'r') as f: 

                    lines = f.readlines() 

                    labels = [int(line.strip().split()[0]) for line in lines] 

                    label = labels[0] 

            else: 

                print(f"Brak etykiety dla obrazu {self.images[index]}") 

 

        if self.transform: 

            image = self.transform(image) 

 

        return image, label 