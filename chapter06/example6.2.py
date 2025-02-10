# 需要安装timm(Torch Image Models)
# pip install timm
from transformers import pipeline

detector = pipeline(task="object-detection")
preds = detector("res/pedestrians-crosswalk.jpg")
print(preds)