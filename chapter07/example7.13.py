import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# 加载大模型
load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
llm_with_tools = llm.bind_tools([multiply])
result = llm_with_tools.invoke("What's the result of 2*3?")
print(result.tool_calls)