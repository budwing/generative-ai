from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn

model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 自动加载分词器
tokenizer = AutoTokenizer.from_pretrained(model)
print(tokenizer.is_fast, type(tokenizer))

# 自动加载模型
model = AutoModelForSequenceClassification.from_pretrained(model)
# 对输入文本分词并生成 input_ids
text = "I like this book."
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
# 执行推理
output = model(**inputs)
print(output)
# 使用softmax将输出结果转换为概率分布
print(nn.functional.softmax(output.logits, dim=1))