from transformers import pipeline

prompt = "The book is super good."
generator = pipeline(task="text-generation")
text = generator(prompt)

print(text)