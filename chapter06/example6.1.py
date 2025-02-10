from transformers import pipeline

classifier = pipeline(task="image-classification")
preds = classifier("res/pedestrians-crosswalk.jpg")
print(*preds, sep="\n")