from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load
import numpy as np

# 加载数据集和分词器
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理
def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)

train_dataset = dataset["train"].map(preprocess, batched=True)
test_dataset = dataset["test"].map(preprocess, batched=True)
print(train_dataset, test_dataset, sep="\n")

small_train_dataset = train_dataset.shuffle(seed=42).select(range(100))
small_eval_dataset = test_dataset.shuffle(seed=42).select(range(10))
small_test_dataset = test_dataset.shuffle(seed=24).select(range(10))
print(small_train_dataset, small_test_dataset, sep="\n")

# 加载一个未经过预训练的模型，随机初始化
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_config(config)

# 设置超参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    eval_strategy="epoch",
)

# 加载评估指标
accuracy = load("accuracy")
f1 = accuracy = load("f1")

# 定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels) | \
        f1.compute(predictions=predictions, references=labels)

# 初始化 Trainer
trainer = Trainer(
    model=model,  # 使用随机初始化的模型
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# 从头开始训练
trainer.train()

# 评估模型
result = trainer.evaluate(eval_dataset=small_test_dataset)
print(result)