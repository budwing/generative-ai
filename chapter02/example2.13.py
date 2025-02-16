# 父类_ScikitCompat
class _ScikitCompat:
    def __init__(self):
        super().__init__()
        print("_ScikitCompat is initializing")
# 父类PushToHubMixin
class PushToHubMixin:
    def __init__(self):
        super().__init__()
        print("PushToHubMixin is initializing")
# 管道类继承自_ScikitCompat, PushToHubMixin
class Pipeline(_ScikitCompat, PushToHubMixin):
    # 构造方法
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    # 调用时执行的方法
    def __call__(self, input:str, *args, **kwds):
        print(f"user input: {input}")
        return "this is a fake implementation"

p = Pipeline("model", "tokenizer")
result = p("my input")
print(result)