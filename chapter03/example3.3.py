# 需要安装sklearn, matplotlib
# pip install scikit-learn matplotlib
from nlp_util import to_embedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 定义需要降维的词汇
words=['king','queen','man','woman','apple','orange','fruit','computer','keyboard','mouse']
word_embeddings=to_embedding(words)

# 使用 t-SNE 将高维词嵌入降维到二维
tsne = TSNE(n_components=2, perplexity=2, random_state=0)
word_embeddings_2d = tsne.fit_transform(word_embeddings)

# 可视化结果
plt.figure(figsize=(10, 6))

for i, word in enumerate(words):
    plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
plt.annotate(word, xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), xytext=(3*len(word), 2), textcoords='offset points', ha='right', va='bottom')

plt.show()