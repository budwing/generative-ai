# 需要安装dotenv, 它可以从.env文件中加载环境变量，大模型服务所需要的token，地址等是通过.env设置的
# pip install python-dotenv
# 需要安装langchain和langchain-openai
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识丰富的AI助手。"),
    ("user", "{question}"),
])
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"question": "美国2020年的总统是谁?"})
print(result)