import torch
import torchvision
from dataset_class import CGAN_dataset
from torch.utils.data import DataLoader
from torch import Tensor

def LSGAN_D(real, fake):
  return (torch.mean((real - 1)**2) + torch.mean(fake**2))

def LSGAN_G(fake):
  return  torch.mean((fake - 1)**2)

def PSNR(img1,img2):
    mse = torch.mean((img1 - img2) ** 2)
    #both image is normalized to range 0-1
    return 20 * torch.log10(1 / torch.sqrt(mse))
      
def get_loaders(
    get_dir,
    batch_size,
    img_transform,
    data_shuffle,
    ):
  data = CGAN_dataset(
        img_dir = get_dir,
        transform = img_transform,
  )
  data_loader = DataLoader(
        data,
        batch_size = batch_size,
        shuffle = data_shuffle,
  )
  return data_loader