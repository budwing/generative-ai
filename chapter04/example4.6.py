from transformers import pipeline

summarizer = pipeline(task="summarization")
sum =summarizer('''
Generative AI, sometimes called gen AI, is artificial intelligence (AI) that can create original content—such as text, images, video, audio or software code—in response to a user's prompt or request.
Generative AI relies on sophisticated machine learning models called deep learning models—algorithms that simulate the learning and decision-making processes of the human brain. 
These models work by identifying and encoding the patterns and relationships in huge amounts of data, and then using that information to understand users' natural language requests or questions and respond with relevant new content.
'''
)

print(sum)