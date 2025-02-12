import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 加载大模型
load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)

# 工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
tools = [multiply]

# 创建模板，只需要包含input和agent_scratchpad变量
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# 创建代理
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
# 在AgentExecutor中执行代理
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
msg = executor.invoke({"input": "what's the result of 2*3*4?"})
print(msg)