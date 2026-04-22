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