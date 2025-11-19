import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.utils import group_norm

ENCODER_CHANNELS = {
    'resnet50': [64, 256, 512, 1024, 2048],
    'resnet101': [64, 256, 512, 1024, 2048],
}

DECODER_SIZE = {
    'default': [512,256,256,256,256,128],
    'large': [512,512,512,512,256,128]
    }



class Encoder(nn.Module):
    def __init__(self, encoder_type, in_channels):
        super().__init__()
        resnet = getattr(models, encoder_type)(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.encoder1(self.maxpool(x0))
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return [x0, x1, x2, x3, x4]

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            group_norm(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class UNetResBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, n_res_blocks):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ResBlock(out_channels) for _ in range(n_res_blocks)])

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.resblocks(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, in_channels, encoder_type, decoder_size, n_res_blocks):
        super().__init__()

        assert decoder_size in DECODER_SIZE
        num_features = DECODER_SIZE[decoder_size]

        assert encoder_type in ENCODER_CHANNELS
        encoder_channels = ENCODER_CHANNELS[encoder_type]

        self.block_0 = UNetResBlock(encoder_channels[-2],encoder_channels[-1], num_features[0], n_res_blocks)
        self.block_1 = UNetResBlock(encoder_channels[-3],num_features[0], num_features[1], n_res_blocks)
        self.block_2 = UNetResBlock(encoder_channels[-4],num_features[1], num_features[2], n_res_blocks)
        self.block_3 = UNetResBlock(encoder_channels[-5],num_features[2], num_features[3], n_res_blocks)      

        self.encoder = Encoder(encoder_type=encoder_type, in_channels=in_channels)

        self.upsample_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_features[3], num_features[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_block_final = ResBlock(num_features[4])

        self.final_conv = nn.Conv2d(num_features[4], num_features[5], kernel_size=1)

    def forward(self, x):

        e1, e2, e3, e4, e5 = self.encoder(x)

        x = self.block_0(e5, e4)
        x = self.block_1(x, e3)
        x = self.block_2(x, e2)
        x = self.block_3(x, e1) 

        x = self.upsample_final(x)
        x = self.res_block_final(x)
        x = self.final_conv(x)

        return x
