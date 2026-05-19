from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence = "The cat sat on the mat"

token_ids = tokenizer.encode(sentence)

print("Token IDs:", token_ids)

for token_id in token_ids:
    word_piece = tokenizer.decode([token_id])
    print(f"  ID {token_id}  →  '{word_piece}'")

print("Total tokens:", len(token_ids))