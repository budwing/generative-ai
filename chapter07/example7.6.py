# 使用RunnableLambda构建最简单的Runnable实例
from langchain_core.runnables import RunnableLambda

runnable1 = RunnableLambda(lambda x: f"in runnable1, got {x}" )
runnable2 = RunnableLambda(lambda x: f"in runnable2, got {x}" )
runnable3 = RunnableLambda(lambda x: f"in runnable3, got {x}" )

chain_par = {
    "key1": runnable1,
    "key2": runnable2,
}
chain = chain_par | runnable3
result = chain.invoke({"xx":"xx"})
print(result)
# chain_par.invoke({"xx":"xx"})