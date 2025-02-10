from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset
from datasets import Audio
import torch

# 加载数据，如果minds14下载不了，请使用本地文件res/input5.3.wav
minds = load_dataset("PolyAI/minds14", name="zh-CN", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16000))

# 加载模型和特征提取器
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name)

# 选取示例
example = minds[0]

# 提取特征
features = processor.feature_extractor(example["audio"]["array"], return_tensors="pt", sampling_rate=16000)
# 前向推理
output = model(**features)
# 解码logits为文本
predicted_ids = torch.argmax(output.logits, dim=-1) 
t = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# 打印结果
print("模型输出:", t[0])
print("样本文本:", example["transcription"])