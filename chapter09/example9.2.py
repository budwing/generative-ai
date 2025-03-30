import evaluate
# 加载召回率
recall_metric = evaluate.load("recall")
results = recall_metric.compute(references=[0, 1, 0, 1, 0], 
                               predictions=[0, 0, 1, 1, 0])
print(results)