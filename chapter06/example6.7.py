from diffusers.models.unets.unet_2d import UNet2DModel


model = UNet2DModel(
    sample_size=28,  # 图像分辨率
    in_channels=1,   # 输入通道数量
    out_channels=1,  # 输出通道数量
    layers_per_block=2,                # 采样块中ResNet的数量
    block_out_channels=(32, 64, 64),  # 采样块通道数量变化
    down_block_types=(
        "DownBlock2D",       # 通用的ResNet下采样块
        "AttnDownBlock2D",  # 自注意力下采样块
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # 自注意力上采样块
        "UpBlock2D",  # 通用ResNet上采样块
    ),

)
print(model)