from landslide.utils import IterableSimpleNamespace

from .segformer import SegformerConfig, SegformerForSemanticSegmentation
from .unet import UNet


def load_model(model: str, data: dict, hyp: IterableSimpleNamespace):
    # Load the model
    out_channels = nc = data.get("nc", 1)
    in_channels = len(data["mean"])
    if model == "unet":
        return
    elif model == "segformer":
        config = SegformerConfig()
        config.num_labels = out_channels
        config.num_channels = in_channels
        return SegformerForSemanticSegmentation(config)
    else:
        return UNet(nc=nc, ch=in_channels)
