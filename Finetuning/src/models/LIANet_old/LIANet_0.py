import torch
import torch.nn as nn
from models.ResUNet import ResUNet
from models.utils import _fast_hash_2d, group_norm


class HashTableEncoder2D(nn.Module):
    """
    2D multi-resolution hash-grid encoder.
    Each level encodes coordinates at a different spatial resolution.
    """
    def __init__(self,
                 levels: int,
                 n_min: int,
                 growth: float,
                 table_size: int,
                 feat_dim: int,
                 table_dtype=torch.float32):
        super().__init__()
        self.levels = int(levels)
        self.n_min = int(n_min)
        self.growth = float(growth)
        self.table_size = int(table_size)
        self.feat_dim = int(feat_dim)

        # Parameter tensor: one hash table per level [L, T, F]
        tables = torch.empty(self.levels, self.table_size, self.feat_dim, dtype=table_dtype)
        nn.init.uniform_(tables, a=-1e-4, b=1e-4)
        self.tables = nn.Parameter(tables)

        # Deterministic per-level seeds for hash decorrelation
        seeds = torch.empty(self.levels, dtype=torch.int64)
        MASK64 = (1 << 64) - 1
        SIGN64 = (1 << 63)
        cA = 0xD6E8FEB86659FD93
        cB = 0x9E3779B97F4A7C15
        for l in range(self.levels):
            u = (cA ^ (l * cB)) & MASK64
            if u >= SIGN64:
                u -= (1 << 64)
            seeds[l] = int(u)
        self.register_buffer("seeds", seeds, persistent=False)

        # Resolution per level (progressively growing grid size)
        Ns = torch.tensor([int(round(self.n_min * (self.growth ** l)))
                           for l in range(self.levels)], dtype=torch.int64)
        self.register_buffer("level_N", Ns, persistent=False)

        self.out_dim = self.levels * self.feat_dim

    @torch.cuda.amp.autocast(False)
    def encode_grid_px_global_batched(self,
                                      x0: torch.Tensor,      # [B]
                                      y0: torch.Tensor,      # [B]
                                      memorized_crop_size: int,
                                      complete_tile_size: int,
                                      ) -> torch.Tensor:
        """
        Encodes a batch of crops into multiscale feature maps.
        Returns: [B, L*F, H, W]
        """
        B = x0.shape[0]
        device = self.tables.device

        # Create a base coordinate grid [H, W] and shift by crop origin (x0, y0)
        xs = torch.linspace(0.5, memorized_crop_size - 0.5, memorized_crop_size,
                            device=device, dtype=torch.float32)
        ys = torch.linspace(0.5, memorized_crop_size - 0.5, memorized_crop_size,
                            device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        px = xx.unsqueeze(0) + x0.to(torch.float32).view(B, 1, 1)
        py = yy.unsqueeze(0) + y0.to(torch.float32).view(B, 1, 1)

        Npts = memorized_crop_size * memorized_crop_size
        pxf = px.reshape(B, Npts)
        pyf = py.reshape(B, Npts)

        feats = []
        for l in range(self.levels):
            Nl = int(self.level_N[l])
            scale = float(Nl) / float(complete_tile_size)
            seed_l = self.seeds[l]

            # Scale coords to level resolution
            xyn_x = pxf * scale
            xyn_y = pyf * scale

            # Integer/fractional parts for bilinear interpolation
            ix0 = torch.floor(xyn_x).to(torch.int64)
            iy0 = torch.floor(xyn_y).to(torch.int64)
            ix1, iy1 = ix0 + 1, iy0 + 1

            fx = (xyn_x - ix0.to(torch.float32)).unsqueeze(-1)
            fy = (xyn_y - iy0.to(torch.float32)).unsqueeze(-1)

            # Hash grid lookup (4 corners per pixel)
            idx00 = _fast_hash_2d(ix0, iy0, seed_l, self.table_size)
            idx10 = _fast_hash_2d(ix1, iy0, seed_l, self.table_size)
            idx01 = _fast_hash_2d(ix0, iy1, seed_l, self.table_size)
            idx11 = _fast_hash_2d(ix1, iy1, seed_l, self.table_size)

            TL = self.tables[l, :, :]  # table for this level

            # Gather feature vectors and interpolate
            f00, f10 = TL[idx00], TL[idx10]
            f01, f11 = TL[idx01], TL[idx11]

            w00 = (1.0 - fx) * (1.0 - fy)
            w10 =  fx        * (1.0 - fy)
            w01 = (1.0 - fx) *  fy
            w11 =  fx        *  fy

            enc_l = (w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11)
            feats.append(enc_l)

        # Concatenate features across levels
        enc = torch.cat(feats, dim=-1)  # [B, N, L*F]
        C = enc.shape[-1]
        enc = enc.view(B, memorized_crop_size, memorized_crop_size, C).permute(0, 3, 1, 2)
        return enc.contiguous()


class LIANet(nn.Module):
    """
    Location Is All You Need (LIANet)
    Encodes spatial coordinates via a hash-grid, adds temporal embedding,
    then passes through a ResUNet to predict dense outputs.
    """
    def __init__(self,
                 timestamp_dim: int,
                 # hash encoder params
                 levels: int,
                 n_min: int,
                 growth: float,
                 table_size: int,
                 feat_dim: int,
                 complete_tile_size: int,
                 # ResUNet params
                 resunet_backbone_size: str,
                 bilinear: bool,
                 # output config
                 out_channels: int,
                 preproj_channels: int | None, 
                 final_activation: str = "ReLU"):
        super().__init__()

        self.complete_tile_size = complete_tile_size
        self.encoder = HashTableEncoder2D(
            levels=levels, n_min=n_min, growth=growth,
            table_size=table_size, feat_dim=feat_dim, table_dtype=torch.float32
        )

        # Learnable temporal embedding (discrete timestamps)
        self.time_emb = nn.Embedding(timestamp_dim, levels * feat_dim)
        nn.init.normal_(self.time_emb.weight, std=1e-3)

        enc_ch = self.encoder.out_dim
        in_ch = enc_ch

        # Optional pre-projection before ResUNet
        if preproj_channels is not None:
            self.preproj = nn.Conv2d(in_ch, preproj_channels, kernel_size=1, bias=True)
            resunet_in = preproj_channels
        else:
            self.preproj = nn.Identity()
            resunet_in = in_ch

        # Backbone network (ResUNet)
        if resunet_backbone_size == "small":
            self.resunet = ResUNet(in_channels=resunet_in, encoder_type="resnet50", decoder_size="default", n_res_blocks=3)
        elif resunet_backbone_size == "large":
            self.resunet = ResUNet(in_channels=resunet_in, encoder_type="resnet101", decoder_size="large", n_res_blocks=3)
        num_channels_last_layer = self.resunet.final_conv.out_channels

        # Output projection head
        self.final_layer = nn.Sequential(
            nn.Conv2d(num_channels_last_layer, num_channels_last_layer // 2, kernel_size=3, padding=1),
            group_norm(num_channels_last_layer // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels_last_layer // 2, out_channels, kernel_size=3, padding=1),
        )

        # Output activation
        if final_activation == "ReLU":
            self.out_act = nn.ReLU(inplace=True)
        elif final_activation in ("none", None):
            self.out_act = nn.Identity()
        else:
            raise ValueError("final_activation must be 'ReLU' or None/'none'")

    @torch.cuda.amp.autocast(False)
    def forward(self,
                timestamps: torch.Tensor,  # [B]
                x0: torch.Tensor,          # [B]
                y0: torch.Tensor,          # [B]
                memorized_crop_size: int = 128,
                ):
        """
        Forward pass: spatial encoding + temporal embedding + ResUNet.
        Returns: [B, out_channels, H, W]
        """
        B = x0.shape[0]
        H = W = memorized_crop_size

        # Hash-based spatial encoding
        enc = self.encoder.encode_grid_px_global_batched(
            x0=x0, y0=y0, memorized_crop_size=memorized_crop_size, complete_tile_size=self.complete_tile_size
        )

        # Add temporal embedding (broadcast over H, W)
        enc += self.time_emb(timestamps).to(torch.float32).unsqueeze(-1).unsqueeze(-1)

        # ResUNet feature extraction and prediction
        x = torch.clamp(enc, -100.0, 100.0)
        x = self.preproj(x) # [B, preproj_channels or enc_ch, H, W]
        y = self.resunet(x) # [B, num_channels_last_layer, H, W]
        y = self.final_layer(y) # [B, out_channels, H, W]
        return self.out_act(y) # [B, out_channels, H, W]
