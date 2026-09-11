from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self._apply_freezing(config)
        self.input_source = config.get("input_source")
        self.output_upsample_scale_factor = config.get("output_upsample_scale_factor")
        self.output_activation = config.get("output_activation", "none")

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

    def _apply_freezing(self, config: Mapping[str, Any]) -> None:
        if config.get("freeze_backbone", False):
            self._freeze_first_existing_module(["encoder", "backbone"])
        if config.get("freeze_decoder", False):
            self._freeze_first_existing_module(["decoder"])
        if config.get("freeze_neck", False):
            self._freeze_first_existing_module(["neck", "necks"])
        if config.get("freeze_head", False):
            self._freeze_first_existing_module(["head", "segmentation_head"])

    def _freeze_first_existing_module(self, names: list[str]) -> None:
        """Freeze the first module matching ``names``.

        Raises if none match: a config asking for a frozen backbone that silently trains it
        instead would invalidate the run without any visible signal.
        """
        for name in names:
            module = getattr(self.model, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
                return
        raise AttributeError(
            f"Cannot freeze {names}: none of these attributes exist on the built model "
            f"(available: {sorted(n for n, _ in self.model.named_children())})."
        )

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        if getattr(self, "input_source", None) in {"alphaearth", "tessera"}:
            x = s2
        else:
            x = self.input_adapter(s2)

        output = self.model(x)
        logits = getattr(output, "output", output)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.ndim != 4:
            raise ValueError(
                "TerraTorch factory model must return logits shaped [B, C, H, W], "
                f"got {tuple(logits.shape)}."
            )
        if self.output_upsample_scale_factor is not None:
            logits = F.interpolate(
                logits,
                scale_factor=float(self.output_upsample_scale_factor),
                mode="bilinear",
                align_corners=False,
            )
        if self.output_activation == "leakyrelu":
            logits = F.leaky_relu(logits)
        elif self.output_activation == "relu":
            logits = F.relu(logits)
        elif self.output_activation == "sigmoid":
            logits = torch.sigmoid(logits)
        elif self.output_activation != "none":
            raise ValueError(f"Unknown output_activation: {self.output_activation}")
        return logits
