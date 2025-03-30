from peft import LoraConfig, TaskType, get_peft_model

model = ... # 加载模型
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, 
                           inference_mode=False, r=8, 
                           lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()