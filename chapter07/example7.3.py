# 需要安装faiss
# pip install faiss-cpu 或者 pip install faiss-gpu
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer
)
import faiss, torch

# 初始化查询编码器及其分词器
q_enc = "facebook/dpr-question_encoder-single-nq-base"
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_enc)
q_encoder = DPRQuestionEncoder.from_pretrained(q_enc)

# 将查询编码为向量
def encode_question(question):
    inputs = q_tokenizer(question, return_tensors='pt',\
                                    truncation=True, max_length=512)
    with torch.no_grad():  # 禁用梯度计算
        embeddings = q_encoder(**inputs).pooler_output
    return embeddings  # 返回 PyTorch 张量

# 示例文档列表
documents = [
    "DPR(Dense Passage Retrieval) is a method for information retrieval.",
    "The transformers library provides pre-trained models for NLP tasks.",
    "Faiss is a library for similarity search and clustering of vectors."
]

# 执行检索
index = faiss.read_index("res/faiss_index/dpr.idx")
query = "What is DPR?"
q_embedding = encode_question(query)
distances, indices = index.search(q_embedding, k=3) # k是最相关文档数量
# 输出检索结果
for idx, distance in zip(indices[0], distances[0]):
    print(f"Document {idx}: {documents[idx]} ({distance})")