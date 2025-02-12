import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# 加载大模型
load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
# ReAct提示模板
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
tools = [multiply]
# 创建ReAct代理
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
# 代理执行环境
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
msg = executor.invoke({"input": "what's the result of 2*3*4?"})
print(msg)