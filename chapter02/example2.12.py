def my_func(param1:str, param2:int, *args, param3:int=200, **kwargs):
    '''带有各种参数形式的函数示例
    Args:
        param1 (str): 字符串参数
        param2 (int): 整型参数
        args: 可变位置参数
        param3 (int, optional): 带默认值的整型参数
        kwargs: 可变关键字参数
    '''
    # 依次打印参数
    print(param1, param2, param3, args, kwargs, sep="\n")

# 按位置传入参数
my_func("hello", 100, 1, 2, x=1, y=2)
# 按关键字传入参数
my_func(param2=200, param1="python", x=1, y=2)