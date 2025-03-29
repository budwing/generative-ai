# 参数剪枝示意性代码
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import prune
# 定义一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 初始化模型
model = SimpleModel()
print("剪枝前的权重：")
print(model.fc1.weight)
# L1权重剪枝
prune.l1_unstructured(model.fc1, name="weight", amount=0.3)  
# 随机剪枝
# prune.random_unstructured(model.fc1, name="weight", amount=0.3)
# 结构化剪枝
# prune.ln_structured(model.fc1, name="weight", n=2, amount=0.3, dim=0)
# 查看剪枝后的权重
print("剪枝后的权重：")
print(model.fc1.weight, model.fc1.weight_orig, model.fc1.weight_mask)

# 恢复参数
prune.remove(model.fc1, name="weight")
# 也可通过手工重置恢复
# model.fc1.weight = nn.Parameter(model.fc1.weight_orig)
print(model.fc1.weight)