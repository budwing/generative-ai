from transformers.pipelines import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert")
# 冻结模型的所有预训练层
for param in model.base_model.parameters():
    param.requires_grad = False