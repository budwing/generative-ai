import torch
import torch.quantization

# 加载预训练模型
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
# 设置模型为评估模式
model.eval()

# 动态量化模型
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化后的模型
torch.save(model_quantized.state_dict(), 'resnet18_quantized_dyn.pth')