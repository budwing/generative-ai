from transformers import (
    RagTokenizer, RagRetriever,
    RagSequenceForGeneration
)
model_id = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(model_id)
retriever = RagRetriever.from_pretrained(model_id, index_name="exact",
                                         use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained(model_id,
                                         retriever=retriever)
q = "how many countries are in europe"
input_dict = tokenizer.prepare_seq2seq_batch(q, return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])