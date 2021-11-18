import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

norm_layer = nn.InstanceNorm2d
class ResBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,in_c,3,1,1),
            norm_layer(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c,in_c,3,1,1)
        )
        self.norm = norm_layer(in_c)
    def forward(self,x):
        return F.relu(self.norm(self.conv(x)+x))

class Generator(nn.Module):
    def __init__(self, in_c, f = 64, blocks = 6):
        super().__init__()
        layers = [
            #padding both side of col, row 3 (512x512->518x518) 
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c,f,7,1,0), norm_layer(f), nn.ReLU(True),
            nn.Conv2d(f,2*f,3,2,1), norm_layer(2*f), nn.ReLU(True),
            nn.Conv2d(2*f,4*f,3,2,1), norm_layer(4*f), nn.ReLU(True),
        ] 
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
        #list.extend for append more than 1 element
        layers.extend([
            #nn.pixelshuffle(2) CxWxH -> C:4xW*2xH*2
            nn.ConvTranspose2d(4*f,4*2*f,3,1,1), nn.PixelShuffle(2), norm_layer(2*f), nn.ReLU(True),
            nn.ConvTranspose2d(2*f,4*f,3,1,1), nn.PixelShuffle(2), norm_layer(f), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(f,in_c,7,1,0),
            nn.Tanh(),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self,in_c,f=64):
        super().__init__()
        self.main = nn.Sequential(
        #1x512x512
        nn.Conv2d(in_c,f,4,2,1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        #64x256x256
        nn.Conv2d(f,f*2,4,2,1, bias=False),
        nn.InstanceNorm2d(f*2),
        nn.LeakyReLU(0.2, inplace=True),
        #128x128x128
        nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(f * 4),
        nn.LeakyReLU(0.2, inplace=True),
        #256x64x64
        nn.Conv2d(f*4,f*8,4,2,1),
        nn.InstanceNorm2d(f*8),
        nn.LeakyReLU(0.2, inplace=True),        
        #512x32x32
        nn.Conv2d(f*8,f*16,4,1,1),
        nn.InstanceNorm2d(f*16),
        nn.LeakyReLU(0.2, inplace=True),
        # 1024x31x31
        nn.Conv2d(f*16,1,4,1,1)
        # 1 x 30 x 30
        )
        
    def forward(self, input):
        return self.main(input)

def test():
    '''
    G = Generator(in_c=1)
    G.to(device = "cuda:0")
    summary(G,(1,512,512))
    '''
    D = Discriminator(in_c=1,f=64)
    D.to(device = "cuda:0")
    summary(D,(1,512,512))

if __name__ == "__main__":
    test()