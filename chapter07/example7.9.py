import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# 初始化模型
load_dotenv()
chat_model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
# 定义 Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template='''
    Based on the context below, answer the question:
    Context: {context}
    Question: {question}
    '''
)
# 创建StuffDocumentsChain
chain = create_stuff_documents_chain(llm=chat_model, prompt=prompt)
# 示例文档
documents = [
    Document(page_content="这是一本关于生成式人工智能的书籍，它是......"),
    Document(page_content="人工智能包括机器学习、神经网络......等。")
]
# 提问
question = "What is the book about?"
# 执行链
result = chain.invoke({"context": documents, "question": question})
print(result)