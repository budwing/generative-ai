from transformers import pipeline

summarizer = pipeline(task="summarization", min_length=50, max_length=80)
sum =summarizer('''
Generative AI, sometimes called gen AI, is artificial intelligence (AI) that can create original content—such as text, images, video, audio or software code—in response to a user's prompt or request.
Generative AI relies on sophisticated machine learning models called deep learning models—algorithms that simulate the learning and decision-making processes of the human brain. 
These models work by identifying and encoding the patterns and relationships in huge amounts of data, and then using that information to understand users' natural language requests or questions and respond with relevant new content.
AI has been a hot technology topic for the past decade, but generative AI, and specifically the arrival of ChatGPT in 2022, has thrust AI into worldwide headlines and launched an unprecedented surge of AI innovation and adoption. 
Generative AI offers enormous productivity benefits for individuals and organizations, and while it also presents very real challenges and risks, businesses are forging ahead, exploring how the technology can improve their internal workflows and enrich their products and services. 
According to research by the management consulting firm McKinsey, one third of organizations are already using generative AI regularly in at least one business function. 
Industry analyst Gartner projects more than 80% of organizations will have deployed generative AI applications or used generative AI application programming interfaces (APIs) by 2026.
'''
)

print(sum)