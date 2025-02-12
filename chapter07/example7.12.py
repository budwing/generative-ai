from langchain.tools import StructuredTool
import asyncio

# 同步函数
def multiply(a:int, b:int)->int:
    """
    Add two integer
    """
    return a*b
# 异步函数
async def amultiply(a:int, b:int)->int:
    """
    Add two integer
    """
    return a*b
# 通过StructuredTool生成工具
calculator = StructuredTool.from_function(
    func=multiply, 
    coroutine=amultiply
)

async def test():
    result = await calculator.ainvoke({"a":12,"b":34})
    return result

print(asyncio.run(test()))
print(calculator.invoke({"a":12,"b":34}))