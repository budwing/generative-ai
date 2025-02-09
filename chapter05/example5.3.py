from transformers import pipeline
from datasets import load_dataset
from datasets import Audio

# 加载minds14数据集
minds = load_dataset("PolyAI/minds14", name="zh-CN", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

# 加载ASR管道
model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
# model = "jonatasgrosman/whisper-large-zh-cv11"
asr = pipeline("automatic-speech-recognition", model=model)

example = minds[0]
o = asr(example["audio"]["array"])

print(o["text"], "\n", example["transcription"])