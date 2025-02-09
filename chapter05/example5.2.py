# 可能需要安装FFmpeg
from transformers import pipeline

classifier = pipeline(task="automatic-speech-recognition")
preds = classifier("/path/to/your/audio")
print(preds)