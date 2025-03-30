# 示意性代码
from datasets import load_dataset
from trl import PPOTrainer

dataset = load_dataset("your/dataset", split="train")
rw = ... # 加载奖励模型

trainer = PPOTrainer(
    model="your/model",
    reward_model= rw,
    train_dataset=dataset,
)
trainer.train()