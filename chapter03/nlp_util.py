import torch
from transformers import AutoTokenizer, AutoModel

def to_embedding(words):
    # 加载预训练的 BERT 模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 获取词汇的输入 ID 和对应的嵌入向量
    word_ids = tokenizer(words, return_tensors='pt', padding=True, truncation=True, max_length=5)['input_ids']
    with torch.no_grad():
        outputs = model(word_ids)
        # 提取 [CLS] 位置的嵌入向量
        word_embeddings = outputs.last_hidden_state[:, 0, :]
    return word_embeddings

    