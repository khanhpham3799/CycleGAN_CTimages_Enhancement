import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c,in_c,3),
            nn.InstanceNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c,in_c,3),
            nn.InstanceNorm2d(in_c)
        )
    def forward(self,x):
        return x+self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_c, f = 64, blocks = 9):
        super().__init__()
        model = [
            #padding both side of col, row 3 (512x512->518x518) 
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c,f,7), nn.InstanceNorm2d(f), nn.ReLU(inplace=True),
            nn.Conv2d(f,2*f,3,2,1), nn.InstanceNorm2d(2*f), nn.ReLU(inplace=True),
            nn.Conv2d(2*f,4*f,3,2,1), nn.InstanceNorm2d(4*f), nn.ReLU(True),
        ] 
        for i in range(int(blocks)):
            model.append(ResBlock(4*f))
        #list.extend for append more than 1 element
        model.extend([
            #nn.pixelshuffle(2) CxWxH -> C:4xW*2xH*2
            nn.ConvTranspose2d(4*f,2*f,3,stride=2,padding=1,output_padding=1), nn.InstanceNorm2d(2*f), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*f,f,3,stride=2,padding=1,output_padding=1), nn.InstanceNorm2d(f), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(f,in_c,7),
            nn.Tanh()
        ])
        self.conv = nn.Sequential(*model)
    
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
        nn.Conv2d(f*4,f*8,4,padding=1),
        nn.InstanceNorm2d(f*8),
        nn.LeakyReLU(0.2, inplace=True),        
        #512x64x64
        nn.Conv2d(f*8,1,4,padding=1)
        # 1 x 62 x 62
        )
        
    def forward(self, x):
        x = self.main(x)
        #calculate average pool with kernel (62,62), 64,1,62,62->64,1,1,1
        x = F.avg_pool2d(x,x.size()[2:])
        #64,1,1,1 -> 64,1 (transform to 1D tensor)
        x = torch.flatten(x,1)
        return x

def test():
    
    G = Generator(in_c=1)
    G.to(device = "cuda:0")
    summary(G,(1,512,512))
    '''
    D = Discriminator(in_c=1,f=64)
    D.to(device = "cuda:0")
    summary(D,(1,512,512))
    '''
if __name__ == "__main__":
    test()