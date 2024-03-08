from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import exposure
from PIL import Image
import cv2
import torch
import numpy as np
import pandas as pd




class BRIXIA(Dataset):
    def __init__(self, filename_list, label_list, label_df, prefix, sum_lung=True,transform=None,augmentation=None,train=True):
        self.filename_list = filename_list
        self.label = label_list
        self.label_df = label_df # pandas dataframe
        self.to_tensor = transforms.ToTensor()
        self.prefix = prefix
        self.train = train
        
        
        self.transform = transform 
        self.augmentation = augmentation 
        
        self.strong_transform = transforms.Compose(
                [
                
                #transforms.ToPILImage(),
                #transforms.RandomHorizontalFlip(), 
                # distortion
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.8),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.8),
                # random sharpness
                transforms.RandomAdjustSharpness(sharpness_factor=0.0,p=0.8),
                
                # random affine
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05,0.05))], p=0.8),
                
                
                ]
            )
        
    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        
        full_filename = self.label_df[self.label_df["Filename"]==self.filename_list[idx]]["Filename"].values[0]
        # full_filename to png
        full_filename = full_filename[:-4]+".png"
        
        file_path = self.prefix + full_filename

        image = Image.open( file_path )
        ##
        
        now_label = torch.tensor(self.label[idx],dtype=torch.int64)
        now_label_float  = torch.tensor(self.label[idx],dtype=torch.float32)
        
        if self.augmentation:
            image=self.augmentation(image=image)
            image=image['image']
        
        image= self.to_tensor(image)
        
        if self.transform: 
            image = self.transform(image).type(torch.float32)
            if self.train:
                image = self.strong_transform(image)            
        
        # 3채널
        if image.shape[0] != 3:
            image = image.expand(3,-1,-1)
        
        return image, now_label,now_label_float, self.filename_list[idx]
    