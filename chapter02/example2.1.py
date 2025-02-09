# 需要安装transformers, sentencepiece, pytorch或tensorflow
# pip install transformers sentencepiece torch
# 推荐安装sacremoses
# pip install sacremoses
# 使用Jupyter执行时，推荐安装ipywidgets
# pip install jupyter ipywidgets
#%%
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
result = translator("我喜欢这套人工智能丛书")
print(result)
# %%
