from datasets import load_dataset
from evaluate import load
from transformers import pipeline

glue_dataset = load_dataset('glue', 'sst2')
glue_metric = load('glue', 'sst2')
sa = pipeline("sentiment-analysis")
tmp = sa(glue_dataset['validation']['sentence'])
predictions = [0 if x['label'] == 'NEGATIVE' else 1 for x in tmp]
results = glue_metric.compute(predictions=predictions, \
    references=glue_dataset['validation']['label'])
print(results)
print(glue_dataset)