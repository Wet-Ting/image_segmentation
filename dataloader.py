from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

rgb2grayWeights = [0.2989, 0.5870, 0.1140]

class myDataset(Dataset):

    @classmethod
    def preprocess(cls, img, scale):
        # w, h = img.size
        # W, H = int(scale * w), int(scale * h)
        img = img.resize((240, 560))
        
        img = np.array(img)
        img = np.power(img/float(np.max(img)), 3)
        img[:, 0:60] = 0
        img[:, 180:240] = 0
        # img = np.power(img/float(np.max(img)), 2)
             
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis = 2)
            
        # HWC -> CHW
        img_trans = img.transpose((2, 0, 1))
        
        #print(img_trans.max())
        
        if img_trans.max() > 255:
            img_trans /= 255
            
        return img_trans

    def __init__(self, imgs_dir, masks_dir, scale = 1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) 
                    if not file.startswith('.')]
                    
    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')
        mask_file = glob(self.masks_dir + idx + '*')      
        
        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])
        
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
                    
    def __len__(self):
        return len(self.ids)