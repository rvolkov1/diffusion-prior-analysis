from diffusers import UNet2DModel
import torch

def create_model(weights_path=None, class_embed_size = 8):
    model = UNet2DModel(
        sample_size=32,
        in_channels=3 + class_embed_size,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),        
    )
    if weights_path!=None:
        print(f"Loading weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path))
    return model

