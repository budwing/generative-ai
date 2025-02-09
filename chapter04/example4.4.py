from transformers import pipeline

text = "I love this book."
translator = pipeline(task="translation_en_to_de")
result = translator(text)

print(result)