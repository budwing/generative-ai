import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# 加载大模型
load_dotenv()
embeddings = AzureOpenAIEmbeddings(
    model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"],
    openai_api_type="azure",
)
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"], 
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
# 加载向量数据库
db = FAISS.load_local("/path/to/faiss_index",  embeddings)
retriever = db.as_retriever()
# 创建提示模板
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template='''
    Based on the context below, answer the question:
    Context: {context}
    Question: {input}
    '''
)
# 创建StuffDocumentsChain
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# 创建RAG链
qa = create_retrieval_chain(retriever, stuff_chain)
# 执行RAG链
query = "your question here?"
result = qa.invoke({"input":query})
print(result)