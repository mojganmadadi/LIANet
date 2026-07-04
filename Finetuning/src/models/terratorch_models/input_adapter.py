from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn


class TerraTorchInputAdapter(nn.Module):
    """Prepare existing LIANet dataloader tensors for TerraTorch factory models."""

    def __init__(
        self,
        dataset_bands: Sequence[str],
        model_bands: Sequence[str],
        means: Optional[Sequence[float]] = None,
        stds: Optional[Sequence[float]] = None,
        scale_factor: float = 1.0,
        offset: float = 0.0,
    ):
        super().__init__()
        self.dataset_bands = list(dataset_bands)
        self.model_bands = list(model_bands)
        self.scale_factor = float(scale_factor)
        self.offset = float(offset)

        self._validate_bands()
        band_to_index: Mapping[str, int] = {
            band: index for index, band in enumerate(self.dataset_bands)
        }
        self.band_indices = [band_to_index[band] for band in self.model_bands]

        if means is None and stds is None:
            self.register_buffer("means", None)
            self.register_buffer("stds", None)
        elif means is None or stds is None:
            raise ValueError("TerraTorch normalization requires both means and stds.")
        else:
            if len(means) != len(self.model_bands) or len(stds) != len(self.model_bands):
                raise ValueError(
                    "TerraTorch means/stds must match the number of model_bands "
                    f"({len(self.model_bands)})."
                )
            self.register_buffer(
                "means", torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
            )
            self.register_buffer(
                "stds", torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1)
            )

    def _validate_bands(self) -> None:
        if not self.dataset_bands:
            raise ValueError("dataset_bands must not be empty.")
        if not self.model_bands:
            raise ValueError("model_bands must not be empty.")
        missing = sorted(set(self.model_bands) - set(self.dataset_bands))
        if missing:
            raise ValueError(
                "model_bands contains bands not present in dataset_bands: "
                + ", ".join(missing)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor [B, C, H, W], got shape {tuple(x.shape)}."
            )
        if x.shape[1] != len(self.dataset_bands):
            raise ValueError(
                f"Expected {len(self.dataset_bands)} input channels from dataset_bands, "
                f"got {x.shape[1]}."
            )

        x = x[:, self.band_indices, :, :]
        x = x * self.scale_factor + self.offset

        if self.means is not None and self.stds is not None:
            x = (x - self.means.to(dtype=x.dtype)) / self.stds.to(dtype=x.dtype)

        return x
