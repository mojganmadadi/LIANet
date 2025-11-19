import yaml
import torch
import inspect
from importlib import import_module

def load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def load_terratorch_model(ckpt_path: str, yaml_path: str):
    """Load a TerraTorch model from YAML + checkpoint."""

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    class_path = model_cfg["class_path"]
    init_args = model_cfg.get("init_args", {})

    ModelClass = load_class(class_path)
    valid_params = set(inspect.signature(ModelClass.__init__).parameters.keys())
    init_args = {k: v for k, v in init_args.items() if k in valid_params}

    model = ModelClass(**init_args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    print(f"Model loaded successfully:\n  - {yaml_path}\n  - {ckpt_path}")

    model.eval()
    return model