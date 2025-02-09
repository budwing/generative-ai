from transformers import AutoTokenizer
# 选择预训练模型的名称
model_name = "bert-base-uncased"
# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 输出分词器的类型
print(tokenizer.tokenize("I love generative AI"))
# 获取BERT词汇表
voc = tokenizer.get_vocab()
# 查看genera和generate在词汇表中的数值ID
print(voc["genera"], voc["generate"])