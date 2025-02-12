# 需要安装faiss
# pip install faiss-cpu 或者 pip install faiss-gpu
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer
)
import faiss, torch

# 初始化文档编码器及其分词器
ctx_enc = "facebook/dpr-ctx_encoder-single-nq-base"
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_enc)
ctx_encoder = DPRContextEncoder.from_pretrained(ctx_enc)

# 将文档编码为向量
def encode_contexts(contexts):
    inputs = ctx_tokenizer(contexts, return_tensors='pt',\
                     padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # 禁用梯度计算
        embeddings = ctx_encoder(**inputs).pooler_output
    return embeddings  # 返回 PyTorch 张量

# 示例文档列表
documents = [
    "DPR(Dense Passage Retrieval) is a method for information retrieval.",
    "The transformers library provides pre-trained models for NLP tasks.",
    "Faiss is a library for similarity search and clustering of vectors."
]

# 创建索引
index = faiss.IndexFlatIP(768)  # 768是DPR模型输出的向量维度
ctx_embeddings = encode_contexts(documents)
index.add(ctx_embeddings)
faiss.write_index(index, "res/faiss_index/dpr.idx")