from typing import Any, Mapping

import torch
import torch.nn as nn

from .input_adapter import TerraTorchInputAdapter


SENTINEL2_L2A_12_BANDS = [
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "WATER_VAPOR",
    "SWIR_1",
    "SWIR_2",
]


def _to_plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


class TerraTorchFactorySegmentationModel(nn.Module):
    """TerraTorch EncoderDecoderFactory model with LIANet input compatibility."""

    def __init__(self, config: Any, num_classes: int):
        super().__init__()
        config = _to_plain_container(config)
        if config is None:
            raise ValueError("Missing terratorch config block.")

        dataset_bands = config.get("dataset_bands", SENTINEL2_L2A_12_BANDS)
        model_bands = config.get("model_bands", dataset_bands)
        normalization = config.get("normalization", {}) or {}
        preprocessing = config.get("preprocessing", {}) or {}

        self.input_adapter = TerraTorchInputAdapter(
            dataset_bands=dataset_bands,
            model_bands=model_bands,
            means=normalization.get("means"),
            stds=normalization.get("stds"),
            scale_factor=preprocessing.get("scale_factor", 1.0),
            offset=preprocessing.get("offset", 0.0),
        )
        self.model = self._build_model(config=config, num_classes=num_classes)

    def _build_model(self, config: Mapping[str, Any], num_classes: int) -> nn.Module:
        try:
            from terratorch.models import EncoderDecoderFactory
        except ImportError as exc:
            raise ImportError(
                "TerraTorch is required for model_type='terratorch_factory'. "
                "Install it in the fine-tuning environment before running this config."
            ) from exc

        task = config.get("task", "segmentation")
        if task != "segmentation":
            raise ValueError(
                "TerraTorchFactorySegmentationModel currently supports only "
                f"task='segmentation', got {task!r}."
            )

        model_args = dict(config.get("model_args", {}) or {})
        configured_num_classes = model_args.get("num_classes")
        if configured_num_classes is not None and configured_num_classes != num_classes:
            raise ValueError(
                "terratorch.model_args.num_classes does not match task num_classes: "
                f"{configured_num_classes} != {num_classes}."
            )
        model_args["num_classes"] = num_classes

        factory = EncoderDecoderFactory()
        return factory.build_model(task=task, **model_args)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(s2)
        output = self.model(x)
        logits = getattr(output, "output", output)
        if logits.ndim != 4:
            raise ValueError(
                "TerraTorch factory model must return logits shaped [B, C, H, W], "
                f"got {tuple(logits.shape)}."
            )
        return logits
