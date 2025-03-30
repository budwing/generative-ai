# 需要安装evaluate库，pip install evaluate
import evaluate
# 加载精确率
precision_metric = evaluate.load("precision")
results = precision_metric.compute(references=[0, 1, 0, 1, 0],
                                  predictions=[0, 0, 1, 1, 0])
print(results)