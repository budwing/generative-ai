from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_config(config)

print(model.encoder.layer[0].attention.self.query.weight)