import evaluate

# 定义预测文本
predictions = ["It is very hot today"]
# 定义参考文本
references = [
    ["It is hot today"]
]
# 加载BLEU
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)