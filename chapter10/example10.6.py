from peft import PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载预训练BERT模型
mn = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(mn)
m = AutoModelForSequenceClassification.from_pretrained(mn, num_labels=5)
# 从路径./peft/checkpoint-4加载PEFT模型
m = PeftModel.from_pretrained(m, "./peft/checkpoint-4")

# 通过PeftConfig和get_peft_model加载PEFT模型
# config = PeftConfig.from_pretrained("./peft/checkpoint-4")
# print(config)
# m = get_peft_model(model, config)
# m.load_adapter("./peft/checkpoint-4", "lora")

m.eval()
inputs = tokenizer("I love this book", return_tensors="pt")
output = m(**inputs)
print(output)