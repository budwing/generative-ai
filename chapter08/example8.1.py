# PyTorch知识蒸馏教程中的核心函数，为了便于排版做了部分修改
# 参考：https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, log_softmax

def train_knowledge_distillation(teacher, student, train_loader,
        epochs, learning_rate, T, soft_weight, ce_weight, device):
    
    optimizer = Adam(student.parameters(), lr=learning_rate)
    teacher.eval()  # 教师模型设置为推理模式
    student.train() # 学生模型设置为训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 教师模型正向传播，不需要做梯度
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            # 学生模型正向传播
            student_logits = student(inputs)
            # 以教师模型输出为标签
            soft_t = softmax(teacher_logits / T, dim=-1)
            soft_prob = log_softmax(student_logits / T, dim=-1)
            # 软标签损失，先计算KL散度（参见式8-7），再做归一化
            kl = torch.sum(soft_t * (soft_t.log() - soft_prob))
            soft_loss = kl / soft_prob.size()[0] * (T**2)
            # 硬标签损失（参见8-8），使用了PyTorch内置函数
            hard_loss = CrossEntropyLoss(student_logits, labels)

            # 软、硬标签损失加权和（参见式8-9）
            loss = soft_weight * soft_loss + ce_weight * hard_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, \
              Loss: {running_loss / len(train_loader)}")