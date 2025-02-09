from transformers import pipeline

classifier = pipeline(task="ner")
preds = classifier("I'm Tian Xuesong from Beijing China.")
print(preds)