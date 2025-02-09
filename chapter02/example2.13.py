# 父类1
class PipelineSuperClass1:
    def __init__(self):
        super().__init__()
        print("super1 is initializing")
# 父类2
class PipelineSuperClass2:
    def __init__(self):
        super().__init__()
        print("super2 is initializing")
# 管道类继承自PipelineSuperClass1, PipelineSuperClass2
class Pipeline(PipelineSuperClass1, PipelineSuperClass2):
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