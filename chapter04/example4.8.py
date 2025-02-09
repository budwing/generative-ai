from transformers import pipeline

text = "The book is <mask> for generative ai."
fill_mask = pipeline(task="fill-mask")
preds = fill_mask(text, top_k=2)
print(preds)