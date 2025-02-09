# 需要安装gensim
# pip install gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
# 示例文本数据
sentences = [
    ["I", "love", "machine", "learning"],
    ["I", "love", "deep", "learning"],
    ["I", "love", "NLP"],
    ["Word2Vec", "is", "a", "great", "tool"],
    ["Gensim", "is", "a", "useful", "library"],
]
# 训练Word2Vec模型
model = Word2Vec(sentences=sentences, vector_size=100, 
                 window=5, min_count=1, workers=4)
# 获取词汇表中的单词
words = list(model.wv.index_to_key)
print("Vocabulary:", words)
# 获取单词向量
vector = model.wv['machine']
print("Vector for 'machine':", vector)
# 找到与给定单词最相似的单词
similar_words = model.wv.most_similar('machine')
print("Words most similar to 'machine':", similar_words)