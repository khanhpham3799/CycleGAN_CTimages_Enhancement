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
from utils import LSGAN_D,LSGAN_G, get_loaders
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim
#writer = SummaryWriter()
in_channel = 1
train_batch_size = 1
test_batch_size = 1

load_model = True
test_run = False
epoch = 0
model_path = "/media/khanhpham/새 볼륨/AAPM_data/checkpoint.pth"
#calculate mean absolute error between each element in (input,target)
criterion_Im = nn.L1Loss()
if torch.cuda.is_available():
    device = 'cuda:1'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')

def train(G_A2B, G_B2A, D_A, D_B, 
        G_A2B_optim, G_B2A_optim, D_A_optim, D_B_optim,
        img_loader, mask_loader):

    loop= tqdm(zip(img_loader,mask_loader))
    G_losses = []
    D_A_losses = []
    D_B_losses = []
    FDL_A2B = []
    FDL_B2A = []
    CL_A = []
    CL_B = []
    ID_B2A = []
    ID_A2B = []
    iters = 0 

    for batch_idx, (img, mask) in enumerate(loop):
        img = img.to(device=device, dtype = torch.float)
        mask = mask.to(device=device, dtype = torch.float)
        
        mask_fake = G_A2B(img)
        img_rec = G_B2A(mask_fake)
        img_fake = G_B2A(mask)
        mask_rec = G_A2B(img_fake)
        
        #Discriminator A
        D_A_optim.zero_grad()

        Disc_loss_A = LSGAN_D(D_A(img), D_A(img_fake.detach()))
        Disc_loss_A.backward()
        D_A_optim.step()
        D_A_losses.append(Disc_loss_A.item())


        #Discriminator B
        D_B_optim.zero_grad()
        Disc_loss_B = LSGAN_D(D_B(mask),D_B(mask_fake.detach()))
        Disc_loss_B.backward()
        D_B_optim.step()
        D_B_losses.append(Disc_loss_B.item())  

        #Generator
        G_A2B_optim.zero_grad()
        G_B2A_optim.zero_grad()

        #Fool discriminator
        Fool_disc_loss_A2B = LSGAN_G(D_B(mask_fake))
        Fool_disc_loss_B2A = LSGAN_G(D_A(img_fake))
        FDL_A2B.append(Fool_disc_loss_A2B)
        FDL_B2A.append(Fool_disc_loss_B2A)

        #Cycle consistency
        Cycle_loss_A = criterion_Im(img_rec,img)*5
        Cycle_loss_B = criterion_Im(mask_rec,mask)*5
        CL_A.append(Cycle_loss_A)
        CL_B.append(Cycle_loss_B)

        #Identity loss
        Id_loss_B2A = criterion_Im(G_B2A(img),img)*10
        Id_loss_A2B = criterion_Im(G_A2B(mask),mask)*10
        ID_B2A.append(Id_loss_B2A)
        ID_A2B.append(Id_loss_A2B)

        #Generator losses
        Loss_G = Fool_disc_loss_A2B + Fool_disc_loss_B2A + Cycle_loss_A +\
                    Cycle_loss_B + Id_loss_A2B + Id_loss_B2A
        Loss_G.backward()
        G_losses.append(Loss_G)
        
        #Optimization step
        G_A2B_optim.step()
        G_B2A_optim.step()

        loop.update(img.shape[0])
        loop.set_postfix({"idx":batch_idx})
        loop.set_description('G:%.4f\tFDL_A2B:%.4f\tFDL_B2A:%.4f\tCL_A:%.4f\tCL_B:%.4f\tID_B2A:%.4f\tID_A2B:%.4f\tLoss_D_A:%.4f\tLoss_D_A:%.4f'
                      % (Loss_G, Fool_disc_loss_A2B, Fool_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,Id_loss_A2B,Disc_loss_A,Disc_loss_B))

    FDL_A2B_t = sum(FDL_A2B)/len(FDL_A2B)
    FDL_B2A_t = sum(FDL_B2A)/len(FDL_B2A)
    CL_A_t = sum(CL_A)/len(CL_A)
    CL_B_t = sum(CL_B)/len(CL_B)
    ID_B2A_t = sum(ID_B2A)/len(ID_B2A)
    ID_A2B_t = sum(ID_A2B)/len(ID_A2B)
    D_A_losses_t = sum(D_A_losses)/len(D_A_losses)
    D_B_losses_t = sum(D_B_losses)/len(D_B_losses)
    G_losses_t = sum(G_losses)/len(G_losses)
    print('G: %.4f\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_A: %.4f'
                      % (G_losses_t, FDL_A2B_t, FDL_B2A_t,CL_A_t,CL_B_t,ID_B2A_t,ID_A2B_t,D_A_losses_t,D_B_losses_t))

    return G_losses_t, FDL_A2B_t, FDL_B2A_t,CL_A_t,CL_B_t,ID_B2A_t,ID_A2B_t,D_A_losses_t,D_B_losses_t

def main():
    learning_rate = 1e-5
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
    G_B2A = Generator(in_c = in_channel).to(device=device)
    D_A = Discriminator(in_c = in_channel).to(device=device)
    D_B = Discriminator(in_c = in_channel).to(device=device)

    G_A2B_optim = torch.optim.Adam(G_A2B.parameters(), lr = learning_rate)
    G_B2A_optim = torch.optim.Adam(G_B2A.parameters(), lr = learning_rate)
    D_A_optim = torch.optim.Adam(D_A.parameters(), lr = learning_rate)
    D_B_optim = torch.optim.Adam(D_B.parameters(), lr = learning_rate)

    if load_model: 
        checkpoint = torch.load(model_path)
        G_A2B.load_state_dict(checkpoint["G_A2B"])
        G_A2B_optim.load_state_dict(checkpoint["G_A2B_optim"])
        G_B2A.load_state_dict(checkpoint["G_B2A"])
        G_B2A_optim.load_state_dict(checkpoint["G_B2A_optim"])
        D_A.load_state_dict(checkpoint["D_A"])
        D_A_optim.load_state_dict(checkpoint["D_A_optim"])
        D_B.load_state_dict(checkpoint["D_B"])
        D_B_optim.load_state_dict(checkpoint["D_B_optim"])
        epoch = checkpoint["epoch"] + 1
        learning_rate = G_A2B_optim.param_groups[0]["lr"]
        print("learning_rate:",learning_rate)
        print("model loaded success!")

    for i in range(epoch, 30):
        if i % 3 == 0:
            learning_rate = 0.1*learning_rate
            G_A2B_optim.param_groups[0]["lr"] = learning_rate
            G_B2A_optim.param_groups[0]["lr"] = learning_rate
            D_A_optim.param_groups[0]["lr"] = learning_rate
            D_B_optim.param_groups[0]["lr"] = learning_rate

        print(f"Epoch: {i}")   
        train_img = get_loaders(
            get_dir = "/media/khanhpham/새 볼륨/AAPM_data/train/quarter_dose/",
            batch_size = train_batch_size,
            img_transform = transform,
            data_shuffle=True,
        )
        train_mask = get_loaders(
            get_dir = "/media/khanhpham/새 볼륨/AAPM_data/train/full_dose/",
            batch_size = train_batch_size,
            img_transform = transform,
            data_shuffle=True,
        )
        G_losses_t, FDL_A2B_t, FDL_B2A_t,CL_A_t,CL_B_t,ID_B2A_t,ID_A2B_t,D_A_losses_t,D_B_losses_t = train(
                                                                                                        G_A2B, G_B2A, D_A, D_B, 
                                                                                                        G_A2B_optim, G_B2A_optim, D_A_optim, D_B_optim,
                                                                                                        train_img, train_mask)

        torch.save({
                    "G_A2B": G_A2B.state_dict(),
                    "G_A2B_optim": G_A2B_optim.state_dict(),
                    "G_B2A": G_B2A.state_dict(),
                    "G_B2A_optim": G_B2A_optim.state_dict(),
                    "D_A": D_A.state_dict(),
                    "D_A_optim": D_A_optim.state_dict(),
                    "D_B": D_B.state_dict(),
                    "D_B_optim": D_B_optim.state_dict(),
                    "epoch": i,
                }, model_path
                )
        print("model successfully saved!")

if __name__ == "__main__":
    main()
