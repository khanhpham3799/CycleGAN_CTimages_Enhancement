import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
model_path = "/media/khanhpham/새 볼륨1/AAPM_data/min_max_get.pth"
get = torch.load(model_path)
max_range = get["max"]
min_range = get["min"]
model_path = "/media/khanhpham/새 볼륨1/AAPM_data/mean_sdt_get.pth"
get = torch.load(model_path)
mean = get["mean"]
std = get["std"]

class CGAN_dataset(Dataset):
    def __init__(self,img_dir,transform=None):
        self.img_link = img_dir
        self.transform = transform
        self.img = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        img_path = os.path.join(self.img_link,self.img[index])
        image = np.load(str(img_path))
        #image = (image-np.min(image))/(np.max(image)-np.min(image))
        image = 1000*(image-0.0194)/0.0194
        image = np.where(image<-1000,-1000,image)
        image = image/4000
        image = image.astype(np.float64)
        if self.transform is not None:
            image = self.transform(image)         
            #image = transforms.functional.normalize(image,mean,std)
            #image = (image-torch.min(image))/(torch.max(image)-torch.min(image))    
        return image
def test():
    img_dir = "/media/khanhpham/새 볼륨/AAPM_data/train/quarter_dose/"
    test = CGAN_dataset(img_dir=img_dir)
    
if __name__ == "__main__":
    test()




