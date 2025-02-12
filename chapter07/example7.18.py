import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 初始化大模型
load_dotenv()
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME_LLM"],
    api_version=os.environ["OPENAI_API_VERSION_LLM"],
)
# 声明图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
# 定义聊天机器人节点，节点接收和返回都是图状态
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 构造状态图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 以流式方式调用图
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        print("exception happened")
        break