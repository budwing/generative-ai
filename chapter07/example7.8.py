# 需要安装langchain-community
# pip install langchain-community
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# 加载嵌入模型
load_dotenv()
embeddings = AzureOpenAIEmbeddings(
    model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"]
)
# 加载PDF文档
loader = PyPDFLoader('/path/to/your/doc.pdf')
document = loader.load()
# 文本拆分
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
# 生成嵌入向量并保存
vs = FAISS.from_documents(texts, embeddings)
vs.save_local("/path/to/faiss_index/")