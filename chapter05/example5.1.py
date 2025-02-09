from transformers import pipeline

# 可能需要安装FFmpeg
classifier = pipeline(task="audio-classification")
preds = classifier("/path/to/your/audio")
print(preds)