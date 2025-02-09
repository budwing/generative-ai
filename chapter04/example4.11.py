from transformers import AutoTokenizer

# 自动加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.is_fast, type(tokenizer))
# 对输入文本分词并生成 input_ids
text = ["I like this book.", "Do you like this book as well?"]
inputs = tokenizer(text, padding=True, truncation=True)
print(inputs)