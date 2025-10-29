import os
from typing import Any, Dict, Type, Union
from urllib.request import urlretrieve

import torch

from landslide.torch_utils import intersect_dicts


# --- Registry Singleton ---
class ModelRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type,
        model_config: Any,
        model_weights: Union[str, None] = None,
    ):
        """Register a model in the registry."""
        if name in cls._registry:
            raise ValueError(f"Model '{name}' is already registered.")
        cls._registry[name] = {
            "class": model_class,
            "config": model_config,
            "weights": model_weights,
        }

    @classmethod
    def get_model_entry(cls, name: str) -> Dict[str, Any]:
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry.")
        return cls._registry[name]


def _ensure_local_weights(path_or_url: str) -> str:
    """Download weights if it's a URL, otherwise verify path exists."""
    if path_or_url.startswith("http"):
        cache_dir = os.path.expanduser("~/.cache/model_registry")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(path_or_url)
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading weights from {path_or_url}...")
            urlretrieve(path_or_url, local_path)
        return local_path
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Weight file not found: {path_or_url}")
    return path_or_url


def load_weights(model: torch.nn.Module, weight_path: str, strict: bool = False) -> torch.nn.Module:
    if weight_path.endswith(".pt") or weight_path.endswith(".pth") or weight_path.endswith(".bin"):
        ckpt = torch.load(weight_path, map_location="cpu")
    elif weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = load_file(weight_path, device="cpu")
    else:
        raise ValueError(f"Unsupported weight file format: {weight_path}")
    
    csd = intersect_dicts(ckpt, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=strict)  # load
    print(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    return model


def load_model(name: str, config: dict = {}, strict: bool = False) -> torch.nn.Module:
    entry = ModelRegistry.get_model_entry(name)
    model_class = entry["class"]
    model_config = entry["config"]
    model_weights = entry.get("weights", None)

    for k, v in config.items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    
    model = model_class(model_config)
    if model_weights is not None:
        weight_path = _ensure_local_weights(model_weights)
        model = load_weights(model, weight_path, strict=strict)

    return model


if __name__ == "__main__":
    from landslide.model.registry import ModelRegistry, load_model
    from landslide.model.segformer import SegformerConfig, SegformerForSemanticSegmentation

    # ModelRegistry.register_model('nvidia/segformer-b0', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")
    model = load_model('nvidia/segformer-b1')