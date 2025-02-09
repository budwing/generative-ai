# 字符串是不可变序列
a = "Hello, Generative AI"
# 使用方括号访问第0个元素，第2~10（不含）个元素
print(a[0], a[2:10], sep="\n")
# 序列通过方括号创建，元素类型可以不相同
b = [1, "AI", 2.3, True, 1]
# 同上
print(b[1], b[2:3], sep="\n")
# 元组
c = (1, "AI", 2.1)
# 使用len计算长度
print(len(c))