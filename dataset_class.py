import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

class CGAN_dataset(Dataset):
    def __init__(self,img_dir,transform=None):
        self.img_link = img_dir
        self.transform = transform
        self.img = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        img_path = os.path.join(self.img_link,self.img[index])
        image = np.load(str(img_path)).astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
            #minmax normalization
            image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
        return image
def test():
    img_dir = "/media/khanhpham/새 볼륨/AAPM_data/train/quarter_dose/"
    test = CGAN_dataset(img_dir=img_dir)
    for i in range(len(test)):
        sample = test[i]
        print(i,sample.shape)

if __name__ == "__main__":
    test()




