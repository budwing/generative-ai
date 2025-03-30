from transformers import AutoTokenizer
from datasets import load_dataset

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