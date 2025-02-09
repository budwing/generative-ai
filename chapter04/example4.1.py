from transformers import pipeline

classifier = pipeline(task="sentiment-analysis")
result = classifier("I like this book.")
print(result)