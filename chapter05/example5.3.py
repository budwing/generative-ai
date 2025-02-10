# 需要安装datasets[audio]
# pip install datasets[audio]
from transformers import pipeline
from datasets import load_dataset
from datasets import Audio
import datasets.config as config

# 加载minds14数据集
minds = load_dataset("PolyAI/minds14", name="zh-CN", trust_remote_code=True, split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16000))

# 加载ASR管道
model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
# model = "jonatasgrosman/whisper-large-zh-cv11"
asr = pipeline("automatic-speech-recognition", model=model)
# 如果minds14下载不了，请使用本地文件
# example = "res/input5.3.wav"
example = minds[0]["audio"]["array"] 
o = asr(example)

print(o["text"], "\n")