import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import os
from models.utils import group_norm
import omegaconf, hydra

def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if sd and all(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd

class DownstreamModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        checkpoint_path_relative: str,
        adaption_strategy: Literal['replace_final_block', "replace_final_block_4x", "replace_final_block_10meter_to_30meter"],
        num_classes: int = None,  # default None (required for regression)
        activation: Literal['none', 'relu'] = 'none',
        # NEW: head capacity knobs
        head_hidden: int = 256,   # try 256 for ~2M-ish; 320/384 for bigger
        head_depth: int = 6,      # increase for more params; 6-8 is typical
        print_head_params: bool = True,
    ):

        super().__init__()

        # -------------------------
        # Load config & pretrained model
        # -------------------------
        config_path = f"{model_path}/used_parameters.json"
        raw_cfg = omegaconf.OmegaConf.load(config_path)
        resolved_dict = omegaconf.OmegaConf.to_container(raw_cfg, resolve=True)
        config = omegaconf.OmegaConf.create(resolved_dict)

        self.generator = hydra.utils.instantiate(config.model)
        ckpt = torch.load(os.path.join(model_path, checkpoint_path_relative), map_location="cpu")
        sd = ckpt["model_state_dict"]
        if all(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        self.generator.load_state_dict(sd, strict=False)

        # -------------------------
        # Check ResUNet structure
        # -------------------------
        print(self.generator)
        # if not hasattr(self.generator, "light_head") or not hasattr(self.generator.light_head, "final_layer"):
        #     raise AttributeError("Expected generator.final_layer to exist.")

        # trunk_out_ch = self.generator.light_head.out_channels
        trunk_out_ch = 128

        if activation == "none":
            self.final_activation = nn.Identity()
        elif activation == "relu":
            self.final_activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.final_activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if adaption_strategy == "replace_final_block":
            self.new_head = nn.Sequential(
                # ---- extra capacity at full width ----
                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                # ---- down to half width (deeper than before) ----
                nn.Conv2d(trunk_out_ch, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                # ---- down to quarter width (extra depth) ----
                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 4, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 4),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 4, trunk_out_ch // 4, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 4),
                nn.ReLU(inplace=True),

                # ---- classifier ----
                nn.Conv2d(trunk_out_ch // 4, num_classes, kernel_size=3, padding=1),
            )

        elif adaption_strategy == "replace_final_block_4x":
            self.new_head = nn.Sequential(
                # ---- extra capacity at full width ----
                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch, trunk_out_ch, kernel_size=3, padding=1),
                group_norm(trunk_out_ch),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

                # ---- half width (deeper than before) ----
                nn.Conv2d(trunk_out_ch, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 2, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 2),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

                # ---- quarter width (extra depth) ----
                nn.Conv2d(trunk_out_ch // 2, trunk_out_ch // 4, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 4),
                nn.ReLU(inplace=True),

                nn.Conv2d(trunk_out_ch // 4, trunk_out_ch // 4, kernel_size=3, padding=1),
                group_norm(trunk_out_ch // 4),
                nn.ReLU(inplace=True),

                # ---- classifier ----
                nn.Conv2d(trunk_out_ch // 4, num_classes, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown adaption_strategy: {adaption_strategy}")

        # Init the weights for new head:
        for m in self.new_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # -------------------------
        # Freeze backbone, keep reconstruction layer
        # -------------------------
        self.generators_final_layer = self.generator.final_layer
        self.generator.final_layer = nn.Identity()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.generator.eval()

        self.num_classes = num_classes  # kept for compatibility

    def forward(self, timestamps: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor):
        with torch.cuda.amp.autocast(True):
            features = self.generator(timestamps, x0, y0)
            reconstruction = self.generators_final_layer(features)
            out = self.new_head(features)
            out = self.final_activation(out)
        return reconstruction, out

# ========== UNet Implementation ==========

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone_size, bilinear=False,
            activation="none",upsample_4x=False):     
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.upsample_4x = upsample_4x

        if backbone_size == "large":
            self.inc = DoubleConv(n_channels, 128)
            self.down1 = Down(128, 256)
            self.down2 = Down(256, 512)
            self.down3 = Down(512, 1024)
            factor = 2 if bilinear else 1
            self.down4 = Down(1024, 2048 // factor)
            self.up1 = Up(2048, 1024 // factor, bilinear)
            self.up2 = Up(1024, 512 // factor, bilinear)
            self.up3 = Up(512, 256 // factor, bilinear)
            self.up4 = Up(256, 128, bilinear)
            if upsample_4x:
                self.post_upsample_conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                                        nn.BatchNorm2d(128),
                                                        nn.ReLU(inplace=True))
            self.outc = OutConv(128, n_classes)

        elif backbone_size == "small":
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            if upsample_4x:
                self.post_upsample_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(inplace=True))
            self.outc = (OutConv(64, n_classes))
        else: 
            raise ValueError("Invalid backbone size. Choose 'large' or 'small'.")
        
        if activation == "none":
            self.final_activation = nn.Identity()
        elif activation == "relu":
            self.final_activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.final_activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.upsample_4x:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            x = self.post_upsample_conv(x)

        logits = self.outc(x)

        return self.final_activation(logits)
    
# ========== Micro UNet Implementation ==========

class DSConv(nn.Module):
    """Depthwise separable conv: DW(3x3) -> BN -> ReLU -> PW(1x1) -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x); x = self.dw_bn(x); x = self.act(x)
        x = self.pw(x); x = self.pw_bn(x); x = self.act(x)
        return x

class DoubleDSConv(nn.Module):
    """Two DSConv blocks, UNet-style"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = DSConv(in_ch, out_ch)
        self.block2 = DSConv(out_ch, out_ch)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class Downmicro(nn.Module):
    """MaxPool -> DoubleDSConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleDSConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Upmicro(nn.Module):
    """Bilinear upsample (or transposed) -> concat skip -> DoubleDSConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # in_ch = channels of (upsampled + skip) concatenated
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # reduce channels after upsample to keep params small before concat
            self.reduce = nn.Conv2d(in_ch // 2, in_ch // 2, 1)  # identity 1x1 to be explicit
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.reduce = nn.Identity()
        self.conv = DoubleDSConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # pad if needed (in case of odd sizes)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class OutConvmicro(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)


class MicroUNet(nn.Module):
    """
    A lightweight UNet variant (~0.5M params @ base≈32) with optional 4× upsample.
    API matches your UNet: (n_channels, n_classes, bilinear=False, activation="none", upsample_4x=False)
    """
    def __init__(self, n_channels, num_classes, bilinear=True, activation="none", upsample_4x=False, base=32):
        super().__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.upsample_4x = upsample_4x

        # Encoder widths: base, 2b, 4b, 8b; bottleneck: 16b (reduced if bilinear)
        # Keep widths small; DSConv keeps params low.
        ch1 = base            # e.g., 32
        ch2 = base * 2        # 64
        ch3 = base * 4        # 128
        ch4 = base * 8        # 256
        factor = 2 if bilinear else 1
        ch5 = base * 16 // factor  # 512 (or 256 if bilinear)

        # inc/down
        self.inc  = DoubleDSConv(n_channels, ch1)
        self.down1 = Downmicro(ch1, ch2)
        self.down2 = Downmicro(ch2, ch3)
        self.down3 = Downmicro(ch3, ch4)
        self.down4 = Downmicro(ch4, ch5)

        # up path (mirror)
        self.up1 = Upmicro(ch5 + ch4, ch4 // factor, bilinear)
        self.up2 = Upmicro((ch4 // factor) + ch3, ch3 // factor, bilinear)
        self.up3 = Upmicro((ch3 // factor) + ch2, ch2 // factor, bilinear)
        self.up4 = Upmicro((ch2 // factor) + ch1, ch1, bilinear)

        # Optional post-upsampling conv (feature space), like your UNet
        if upsample_4x:
            self.post_upsample_conv = nn.Sequential(
                nn.Conv2d(ch1, ch1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch1),
                nn.ReLU(inplace=True),
            )

        self.outc = OutConvmicro(ch1, num_classes)

        # final activation
        if activation == "none":
            self.final_activation = nn.Identity()
        elif activation == "relu":
            self.final_activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.final_activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.upsample_4x:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            x = self.post_upsample_conv(x)

        logits = self.outc(x)
        return self.final_activation(logits)
