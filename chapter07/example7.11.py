from langchain_core.tools import tool

@tool
def add(a:int, b:int)->int:
    """
    Add two integer
    """
    return a+b

print(add.name)
print(add.description)
print(add.args)
print(add.invoke({
    "a":12,
    "b":34
}))