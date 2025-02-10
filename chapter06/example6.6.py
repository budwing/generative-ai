from torch.nn import (
    Module, ModuleList, Conv2d,
    SiLU, MaxPool2d, Upsample
)

class BasicUNet(Module):
    """最简UNet实现"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = ModuleList(
            [
                Conv2d(in_channels, 32, kernel_size=5, padding=2),
                Conv2d(32, 64, kernel_size=5, padding=2),
                Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = ModuleList(
            [
                Conv2d(64, 64, kernel_size=5, padding=2),
                Conv2d(64, 32, kernel_size=5, padding=2),
                Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = SiLU()  # The activation function
        self.downscale = MaxPool2d(2)
        self.upscale = Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  
            if i < 2:  
                h.append(x)  
                x = self.downscale(x)  

        for i, l in enumerate(self.up_layers):
            if i > 0: 
                x = self.upscale(x)  # Upscale
                x += h.pop()  
            x = self.act(l(x))  

        return x