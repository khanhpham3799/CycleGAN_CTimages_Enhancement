import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from CGAN_model import Discriminator, Generator
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import LSGAN_D,LSGAN_G, get_loaders,PSNR
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim

writer = SummaryWriter()
in_channel = 1
test_batch_size = 1

model_path = "/media/khanhpham/새 볼륨/AAPM_data/checkpoint.pth"

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')

def test(G_A2B, img_loader, mask_loader):
    PSNR_list = []
    SSIM_list = []
    loop= tqdm(zip(img_loader,mask_loader))
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(loop):
            img = img.to(device=device, dtype = torch.float)
            mask = mask.to(device=device, dtype = torch.float)
            mask_fake = G_A2B(img)
            writer.add_images("img", img, batch_idx)
            writer.add_images("mask_img/gt", mask, batch_idx)
            writer.add_images("mask_img/pred", mask_fake, batch_idx)
            PSNR_score = PSNR(mask_fake, mask)
            SSIM_score = ssim(mask_fake, mask, data_range=1, size_average=False) 
            PSNR_list.append(PSNR_score)
            SSIM_list.append(SSIM_score)
        
            loop.update(img.shape[0])
            loop.set_postfix({"idx":batch_idx})
            loop.set_description("PSNR_test:%.5f|SSIM_test:%.5f"%(PSNR_score,SSIM_score))
    return sum(PSNR_list)/len(PSNR_list),sum(SSIM_list)/len(SSIM_list)

def main():
    learning_rate = 1e-3
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ])
    
    test_img = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/AAPM_data/test/quarter_dose/",
        batch_size = test_batch_size,
        img_transform = transform,
        data_shuffle=False,
    )
    test_mask = get_loaders(
        get_dir = "/media/khanhpham/새 볼륨/AAPM_data/test/full_dose/",
        batch_size = test_batch_size,
        img_transform = transform,
        data_shuffle=False,
    )

    G_A2B = Generator(in_c = in_channel).to(device=device)
    G_A2B_optim = torch.optim.Adam(G_A2B.parameters(), lr = learning_rate)
    checkpoint = torch.load(model_path)
    G_A2B.load_state_dict(checkpoint["G_A2B"])
    G_A2B_optim.load_state_dict(checkpoint["G_A2B_optim"])

    PSNR_new, SSIM = test(G_A2B, test_img, test_mask)
    print(f"Test PSNR: {PSNR_new}|SSIM: {SSIM}") 
    writer.flush()
if __name__ == "__main__":
    main()
