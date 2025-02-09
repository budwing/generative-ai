from transformers import pipeline

text = "我喜欢这本书."
translator = pipeline(task="translation", model="facebook/m2m100_418M", 
src_lang="zh", tgt_lang="de")
result = translator(text)

print(result)