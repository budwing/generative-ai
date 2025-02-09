from transformers import pipeline

text = ["我喜欢这本书", "这是我见过最好的人工智能书籍"]
translator = pipeline(task="translation", model="facebook/m2m100_418M",
                      src_lang="zh", tgt_lang="en")
result = translator(text)
print(result)