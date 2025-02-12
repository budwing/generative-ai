from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer
)
import torch

# 初始化文档编码器及其分词器
ctx_enc = "facebook/dpr-ctx_encoder-single-nq-base"
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_enc)
ctx_encoder = DPRContextEncoder.from_pretrained(ctx_enc)

# 初始化查询编码器及其分词器
q_enc = "facebook/dpr-question_encoder-single-nq-base"
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_enc)
q_encoder = DPRQuestionEncoder.from_pretrained(q_enc)

# 示例文档列表
documents = [
    "DPR(Dense Passage Retrieval) is a method for information retrieval.",
    "The transformers library provides pre-trained models for NLP tasks.",
    "Faiss is a library for similarity search and clustering of vectors."
]

# 将文档编码为向量
def encode_contexts(contexts):
    inputs = ctx_tokenizer(contexts, return_tensors='pt',\
                     padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # 禁用梯度计算
        embeddings = ctx_encoder(**inputs).pooler_output
    return embeddings  # 返回 PyTorch 张量

# 将查询编码为向量
def encode_question(question):
    inputs = q_tokenizer(question, return_tensors='pt',\
                                    truncation=True, max_length=512)
    with torch.no_grad():  # 禁用梯度计算
        embeddings = q_encoder(**inputs).pooler_output
    return embeddings  # 返回 PyTorch 张量

# 计算内积相似度
def inner_product_similarity(query_embedding, ctx_embeddings):
    query_embedding = query_embedding.squeeze()  # 去掉批次维度
    # 计算所有文档向量与查询向量的内积
    similarities = torch.matmul(ctx_embeddings, query_embedding)
    return similarities.tolist()

# 创建文档向量
ctx_embeddings = encode_contexts(documents)

# 执行检索
query = "What is DPR?"
query_embedding = encode_question(query)
# 计算相似度
similarities = inner_product_similarity(query_embedding, ctx_embeddings)
# 获取最相关的文档
top_k = 3
tops = sorted(range(len(similarities)),\
               key=lambda i: similarities[i], reverse=True)[:top_k]

# 输出检索结果
for i in tops:
    print(f"Doc{i}: {documents[i]} (Similarity: {similarities[i]:.4f})")