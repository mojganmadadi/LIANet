import torch
import torch.nn as nn
from models.ResUNet import ResUNet
from models.utils import _fast_hash_2d, group_norm
# from models.Unet import UNet

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
                 bilinear: bool = True,
                 vectorized: bool = True,
                 table_dtype=torch.float32):
        super().__init__()
        self.levels = int(levels)
        self.n_min = int(n_min)
        self.growth = float(growth)
        self.table_size = int(table_size)
        self.feat_dim = int(feat_dim)
        self.vectorized = bool(vectorized)
        self.bilinear = bool(bilinear)

        tables = torch.empty(self.table_size, self.feat_dim, dtype=table_dtype)
        nn.init.uniform_(tables, a=-1e-2, b=1e-2)
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
        # self.register_buffer("level_N", Ns, persistent=False)
        self.register_buffer("level_N_f", Ns.to(torch.float32), persistent=False)
        self.out_dim = self.levels * self.feat_dim
        self._base_xx = None
        self._base_yy = None
        self._cached_hw = None


    def _get_base_grid(self, H: int, device: torch.device):
        # Rebuild only if size changed or buffer not initialized or device changed
        if (self._cached_hw != H or self._base_xx.numel() == 0 or self._base_xx.device != device):
            xs = torch.arange(H, device=device, dtype=torch.float32)
            ys = torch.arange(H, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self._base_xx = xx  # [H, W]
            self._base_yy = yy
            self._cached_hw = H
        return self._base_xx, self._base_yy

    # @torch.cuda.amp.autocast(True)
    # def encode_grid_px_global_batched(self,
    #                                   x0: torch.Tensor,      # [B]
    #                                   y0: torch.Tensor,      # [B]
    #                                   memorized_crop_size: int,
    #                                   complete_tile_size: int,
    #                                   ) -> torch.Tensor:
    #     """
    #     Encodes a batch of crops into multiscale feature maps.
    #     Returns: [B, L*F, H, W]
    #     """
    #     B = x0.shape[0]
    #     device = self.tables.device

    #     xx, yy = self._get_base_grid(memorized_crop_size, device)
    #     px = xx.unsqueeze(0) + x0.to(torch.float32).view(B, 1, 1)
    #     py = yy.unsqueeze(0) + y0.to(torch.float32).view(B, 1, 1)

    #     Npts = memorized_crop_size * memorized_crop_size
    #     pxf = px.reshape(B, Npts)
    #     pyf = py.reshape(B, Npts)

    #     feats = []
    #     for l in range(self.levels):
    #         Nl = self.level_N[l]
    #         scale = float(Nl) / float(complete_tile_size)
    #         seed_l = self.seeds[l]

    #         # Scale coords to level resolution
    #         xyn_x = pxf * scale
    #         xyn_y = pyf * scale

    #         # Integer/fractional parts for bilinear interpolation
    #         ix0 = torch.floor(xyn_x).to(torch.int64)
    #         iy0 = torch.floor(xyn_y).to(torch.int64)
    #         ix1, iy1 = ix0 + 1, iy0 + 1

    #         fx = (xyn_x - ix0.to(torch.float32)).unsqueeze(-1)
    #         fy = (xyn_y - iy0.to(torch.float32)).unsqueeze(-1)

    #         # Hash grid lookup (4 corners per pixel)
    #         idx00 = _fast_hash_2d(ix0, iy0, seed_l, self.table_size)
    #         idx10 = _fast_hash_2d(ix1, iy0, seed_l, self.table_size)
    #         idx01 = _fast_hash_2d(ix0, iy1, seed_l, self.table_size)
    #         idx11 = _fast_hash_2d(ix1, iy1, seed_l, self.table_size)

    #         # TL = self.tables[l, :, :]  # table for this level
    #         TL = self.tables  # shared table for all levels

    #         # Gather feature vectors and interpolate
    #         f00, f10 = TL[idx00], TL[idx10]
    #         f01, f11 = TL[idx01], TL[idx11]

    #         w00 = (1.0 - fx) * (1.0 - fy)
    #         w10 =  fx        * (1.0 - fy)
    #         w01 = (1.0 - fx) *  fy
    #         w11 =  fx        *  fy

    #         enc_l = (w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11)
    #         feats.append(enc_l)
        
    #     # Concatenate features across levels
    #     enc = torch.cat(feats, dim=-1)  # [B, N, L*F]
    #     C = enc.shape[-1]
    #     enc = enc.view(B, memorized_crop_size, memorized_crop_size, C).permute(0, 3, 1, 2)
    #     return enc.contiguous()
    @torch.cuda.amp.autocast(True)
    def encode_grid_px_global_batched(
        self,
        x0: torch.Tensor,      # [B]
        y0: torch.Tensor,      # [B]
        memorized_crop_size: int,
        complete_tile_size: int,
    ) -> torch.Tensor:
        """
        Vectorized over levels.
        Returns: [B, L*F, H, W]
        """
        B = x0.shape[0]
        device = self.tables.device
        H = W = memorized_crop_size
        L = self.levels
        F = self.feat_dim

        # --- base grid (cached) ---
        xx, yy = self._get_base_grid(H, device)          # [H, W] float32
        # shift by crop origin
        # px = xx.unsqueeze(0) + x0.to(torch.float32).view(B, 1, 1)  # [B, H, W]
        # py = yy.unsqueeze(0) + y0.to(torch.float32).view(B, 1, 1)  # [B, H, W]
        px = xx.unsqueeze(0) + x0.view(B, 1, 1)
        py = yy.unsqueeze(0) + y0.view(B, 1, 1)
        
        N = H * W
        pxf = px.reshape(B, N)  # [B, N]
        pyf = py.reshape(B, N)  # [B, N]

        # --- scales per level: [L] float32 ---
        # scale = Nl / complete_tile_size
        scales = (self.level_N_f / float(complete_tile_size))  # [L]

        TL = self.tables  # [T, F]

        if self.vectorized:
            # --- broadcast coords to [B, L, N] ---
            # xyn_x[b,l,n] = pxf[b,n] * scales[l]
            xyn_x = pxf[:, None, :] * scales[None, :, None]  # [B, L, N]
            xyn_y = pyf[:, None, :] * scales[None, :, None]  # [B, L, N]

            # integer/fractional parts
            ix0 = torch.floor(xyn_x).to(torch.int64)  # [B, L, N]
            iy0 = torch.floor(xyn_y).to(torch.int64)  # [B, L, N]
            ix1 = ix0 + 1
            iy1 = iy0 + 1

            fx = (xyn_x - ix0).unsqueeze(-1)  # [B, L, N, 1]
            fy = (xyn_y - iy0).unsqueeze(-1)  # [B, L, N, 1]
            # --- seeds per level, broadcast to [B, L, N] ---
            seed_bln = self.seeds[None, :, None].expand(B, L, N)  # [B, L, N]

            # flatten everything to 1D for hashing + indexing
            ix0f = ix0.reshape(-1)
            iy0f = iy0.reshape(-1)
            ix1f = ix1.reshape(-1)
            iy1f = iy1.reshape(-1)
            seedf = seed_bln.reshape(-1)

            # hash indices (flattened): [B*L*N]
            idx00 = _fast_hash_2d(ix0f, iy0f, seedf, self.table_size)
            idx10 = _fast_hash_2d(ix1f, iy0f, seedf, self.table_size)
            idx01 = _fast_hash_2d(ix0f, iy1f, seedf, self.table_size)
            idx11 = _fast_hash_2d(ix1f, iy1f, seedf, self.table_size)

            # gather features: [BLN, F]
            f00 = TL[idx00]
            f10 = TL[idx10]
            f01 = TL[idx01]
            f11 = TL[idx11]

            if self.bilinear:
                # reshape weights to match gathered features
                fx_f = fx.reshape(-1, 1)  # [BLN, 1]
                fy_f = fy.reshape(-1, 1)  # [BLN, 1]

                w00 = (1.0 - fx_f) * (1.0 - fy_f)
                w10 = fx_f * (1.0 - fy_f)
                w01 = (1.0 - fx_f) * fy_f
                w11 = fx_f * fy_f

                enc_flat = (w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11)  # [BLN, F]
            else:
                enc_flat = 0.25 * (f00 + f10 + f01 + f11)  # [BLN, F]

            # unflatten back to [B, L, N, F]
            enc = enc_flat.view(B, L, N, F)
        else:
            feats = []
            for l in range(L):
                scale = scales[l]
                xyn_x = pxf * scale  # [B, N]
                xyn_y = pyf * scale  # [B, N]

                ix0 = torch.floor(xyn_x).to(torch.int64)
                iy0 = torch.floor(xyn_y).to(torch.int64)
                ix1 = ix0 + 1
                iy1 = iy0 + 1

                seed_l = self.seeds[l]
                idx00 = _fast_hash_2d(ix0, iy0, seed_l, self.table_size)
                idx10 = _fast_hash_2d(ix1, iy0, seed_l, self.table_size)
                idx01 = _fast_hash_2d(ix0, iy1, seed_l, self.table_size)
                idx11 = _fast_hash_2d(ix1, iy1, seed_l, self.table_size)

                f00 = TL[idx00]
                f10 = TL[idx10]
                f01 = TL[idx01]
                f11 = TL[idx11]
                feats.append(0.25 * (f00 + f10 + f01 + f11))  # [B, N, F]

            enc = torch.cat(feats, dim=-1)
            enc = enc.view(B, N, L, F)

        # reorder to [B, N, L*F]
        enc = enc.permute(0, 2, 1, 3).contiguous().view(B, N, L * F)

        # reshape to [B, L*F, H, W]
        enc = enc.view(B, H, W, L * F).permute(0, 3, 1, 2).contiguous()
        return enc



class ConvResBlock(nn.Module):
    def __init__(self, ch, groups=32, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            group_norm(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            group_norm(ch),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.res_scale * self.block(x))

class LiteCNNHead(nn.Module):
    """Very light head: 1x1 reduce -> a couple residual 3x3 -> 1x1 out."""
    def __init__(self, in_ch, out_ch, hidden=256, n_blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            group_norm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvResBlock(hidden) for _ in range(n_blocks)])
        self.proj = nn.Conv2d(hidden, out_ch, 1, bias=True)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.proj(x)


class BaseTimeEncoder(nn.Module):
    """Abstract: map time input -> [B, out_dim]"""
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        

class IndexTimeEncoder(BaseTimeEncoder):
    """
    Learned embedding for discrete time indices.
    t: LongTensor [B]
    """
    def __init__(self, num_timestamps: int, out_dim: int):
        super().__init__(out_dim)
        self.emb = nn.Embedding(num_timestamps, out_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] long
        return self.emb(t)  # [B, out_dim]


class SinusoidalTimeEncoder1D(BaseTimeEncoder):
    """
    Fixed (non-learned) sinusoidal encoding (like Transformer),
    followed by a linear projection to out_dim.

    t: float [B] (e.g. days since reference), reasonably scaled.
    """
    def __init__(self, num_frequencies: int, out_dim: int):
        super().__init__(out_dim)
        self.num_frequencies = num_frequencies
        # frequencies: 1, 2, 4, ..., 2^{L-1}
        freqs = 2 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

        in_dim = 2 * num_frequencies  # sin+cos
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float
        t = t.view(-1, 1)  # [B, 1]
        angles = t * self.freqs[None, :]  # [B, F]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        enc = torch.cat([sin, cos], dim=-1)  # [B, 2F]
        return self.proj(enc)  # [B, out_dim]


class LearnedFourierTimeEncoder1D(BaseTimeEncoder):
    """
    Learned Fourier features: frequencies are learnable.
    t: float [B]
    """
    def __init__(self, num_frequencies: int, out_dim: int):
        super().__init__(out_dim)
        self.num_frequencies = num_frequencies
        # learnable frequencies
        self.freqs = nn.Parameter(
            torch.randn(num_frequencies, dtype=torch.float32)
        )

        in_dim = 2 * num_frequencies
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float
        t = t.view(-1, 1)  # [B, 1]
        angles = t * self.freqs[None, :]  # [B, F]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        enc = torch.cat([sin, cos], dim=-1)  # [B, 2F]
        return self.proj(enc)  # [B, out_dim]


class MLPTimeEncoder(BaseTimeEncoder):
    """
    "Complex" time encoder: multi-dimensional time input
    (e.g. [t_days, doy_sin, doy_cos, ...]) -> MLP -> [B, out_dim].

    t: float [B, time_input_dim]
    """
    def __init__(self,
                 time_input_dim: int,
                 out_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2):
        super().__init__(out_dim)

        layers = []
        in_dim = time_input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

        # mild init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, time_input_dim]
        return self.net(t)  # [B, out_dim]


class LIANetLight(nn.Module):
    """
    Location Is All You Need (LIANet)
    Encodes spatial coordinates via a hash-grid, adds temporal embedding,
    then passes through a ResUNet to predict dense outputs.
    """
    def __init__(self,
                 # time config
                 timestamp_dim: int,          # used for 'index' mode
                 time_mode: str,    # 'index', 'sinusoidal', 'fourier_learned', 'mlp'
                 num_time_frequencies: int, #8
                 time_input_dim: int | None,  # needed for 'mlp'
                 time_mlp_hidden: int, # 64
                 # hash encoder params
                 levels: int,
                 n_min: int,
                 growth: float,
                 table_size: int,
                 feat_dim: int,
                 complete_tile_size: int,
                 out_channels: int,
                 preproj_channels: int | None, 
                 resunet_backbone_size: str = "small",
                 bilinear: bool = True,
                 hash_vectorized: bool = True,
                 final_activation: str = None,
                 n_blocks: int = 3
                 ):
        super().__init__()

        self.complete_tile_size = complete_tile_size
        self.time_mode = time_mode
        self.n_blocks = n_blocks

        self.encoder = HashTableEncoder2D(
            levels=levels, n_min=n_min, growth=growth,
            table_size=table_size, feat_dim=feat_dim, vectorized=hash_vectorized,
            table_dtype=torch.float32
        )
        enc_ch = self.encoder.out_dim  # = levels * feat_dim
        # Learnable temporal embedding (discrete timestamps)
        # self.time_emb = nn.Embedding(timestamp_dim, levels * feat_dim)
        # nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)

        # --- time encoder selection ---
        if time_mode == "index":
            # discrete timestamps: t is LongTensor [B] with values in [0, timestamp_dim)
            self.time_encoder = IndexTimeEncoder(num_timestamps=timestamp_dim,
                                                 out_dim=enc_ch)

        elif time_mode == "sinusoidal":
            # t is float [B], e.g. days since reference, reasonably scaled
            self.time_encoder = SinusoidalTimeEncoder1D(
                num_frequencies=num_time_frequencies,
                out_dim=enc_ch
            )

        elif time_mode == "fourier_learned":
            # t is float [B]
            self.time_encoder = LearnedFourierTimeEncoder1D(
                num_frequencies=num_time_frequencies,
                out_dim=enc_ch
            )

        elif time_mode == "mlp":
            # t is float [B, time_input_dim] (e.g. [t_days, doy_sin, doy_cos])
            if time_input_dim is None:
                raise ValueError("time_input_dim must be set for time_mode='mlp'")
            self.time_encoder = MLPTimeEncoder(
                time_input_dim=time_input_dim,
                out_dim=enc_ch,
                hidden_dim=time_mlp_hidden,
                num_layers=3,
            )
        else:
            raise ValueError(f"Unknown time_mode: {time_mode}")


        # Optional pre-projection before ResUNet
        if preproj_channels is not None:
            self.preproj = nn.Conv2d(enc_ch, preproj_channels, kernel_size=1, bias=True)
            resunet_in = preproj_channels
        else:
            self.preproj = nn.Identity()
            resunet_in = enc_ch

        # self.light_head = ResUNet(in_channels=resunet_in, encoder_type="resnet50", decoder_size="default", n_res_blocks=3)
        self.light_head = LiteCNNHead(in_ch=resunet_in, out_ch=128, hidden=256, n_blocks=self.n_blocks)
        num_channels_last_layer = 128

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

    @torch.cuda.amp.autocast(True)
    def forward(self,
                timestamps: torch.Tensor,  # [B]
                x0: torch.Tensor,          # [B]
                y0: torch.Tensor,          # [B]
                memorized_crop_size: int = 128,
                ):
        """
        Forward pass: spatial encoding + temporal embedding + CNN head.
        Returns: [B, out_channels, H, W]

        timestamps:
          - time_mode='index':       Long [B] with values in [0, timestamp_dim)
          - time_mode='sinusoidal':  Float [B], e.g. days since reference
          - time_mode='fourier_learned': Float [B], same as above
          - time_mode='mlp':         Float [B, time_input_dim]
        """
        B = x0.shape[0]
        H = W = memorized_crop_size

        # Hash-based spatial encoding
        enc = self.encoder.encode_grid_px_global_batched(
            x0=x0, y0=y0, memorized_crop_size=memorized_crop_size, complete_tile_size=self.complete_tile_size
        )

                # --- temporal encoding ---
        t_feat = self.time_encoder(timestamps).to(torch.float32)  # [B, enc_ch]
        t_feat = t_feat.unsqueeze(-1).unsqueeze(-1)               # [B, enc_ch, 1, 1]

        # broadcast and add
        enc = enc + t_feat


        x = self.preproj(enc) # [B, preproj_channels or enc_ch, H, W]
        y = self.light_head(x) # [B, num_channels_last_layer, H, W]
        y = self.final_layer(y) # [B, out_channels, H, W]
        return self.out_act(y) # [B, out_channels, H, W]
