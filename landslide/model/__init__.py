from landslide.model.registry import ModelRegistry, load_model
from landslide.model.segformer import SegformerConfig, SegformerForSemanticSegmentation
from landslide.model.unet import UNet, UNetConfig

ModelRegistry.register_model('nvidia/segformer-b0', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")
ModelRegistry.register_model('nvidia/segformer-b1', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")
ModelRegistry.register_model('nvidia/segformer-b2', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b3', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b4', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b5', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('L4S/unet', UNet, UNetConfig(num_labels=1, num_channels=3))
