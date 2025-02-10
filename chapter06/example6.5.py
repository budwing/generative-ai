# 需要安装diffusers
# pip install diffusers
import torch
from PIL import Image
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline
import time

start = time.time()

# 设置执行设备：mps,gpu或cpu
device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() else "cpu"
# 加载预训练好的stable diffusion模型
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# 设置随机数种子，用以保证每次生成的结果相同
generator = torch.Generator(device=device).manual_seed(42)

# 执行管道，生成图像
pipe_output = pipe(
    prompt="Oil painting of an autumn cityscape",  # 正向提示
    negative_prompt="Oversaturated, blurry, low quality", # 逆向提示
    height=480,
    width=640,  # 图像尺寸
    guidance_scale=8,  # 遵从提示的强度
    num_inference_steps=35,  # 生成图像的步数
    generator=generator,  # 使用固定的随机数
)

# 保存图片
image = pipe_output.images[0]
image.save("res/autumn_oil_painting.png")
print(time.time()-start)