from transformers import pipeline

qa = pipeline("question-answering")
result = qa(question="What's the capital of USA?", context='''
        The United States, with its capital in Washington, D.C., 
        is a federal republic known for diversity, 
        innovation, and global influence.''')
print(result)