# 序列推导式：生成0~9的平方
l1 = [x**2 for x in range(10)]
print(l1)
# 字典推导式：以整数1为例，键为字符串num1，值为整型1
d1 = {f"num{n}":n for n in range(10)}
print(d1)
# 集合推导式：生成偶数集合，是典型的for与if相结合的用法
even_list = {x for x in range(10) if x%2==0}
print(even_list)
# 生成even,odd序列，另一种for与if相结合的方式
even_or_odd = ["even" if x%2==0 else "odd" for x in range(10)]
print(even_or_odd)
# 元组序列的推导式
tuple_list = [(k,v) for k,v in d1.items()]
print(tuple_list)
# 下面的式子并不是元组推导式，它生成的是生成器
generator = (x*2 for x in range(5))
print(generator)
# 生成器在迭代时才生成数值
for num in generator:
    print(num, end=",")