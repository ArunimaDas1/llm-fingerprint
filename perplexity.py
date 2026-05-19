import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading models...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

distil_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distil_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
distil_model.eval()

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    return torch.exp(outputs.loss).item()

# These are calibration sentences — totally neutral, neither AI nor human
# We use these to find each model's "normal" level of surprise
calibration_texts = [
    "The weather today is sunny and warm",
    "She walked to the store to buy milk",
    "The dog ran across the green field",
    "He opened the book and started reading",
    "They sat together and had dinner",
]

print("Calibrating models (finding their normal surprise level)...")

# Find GPT-2's average perplexity on neutral text
gpt2_baseline = sum(
    compute_perplexity(t, gpt2_tokenizer, gpt2_model) 
    for t in calibration_texts
) / len(calibration_texts)

# Find DistilGPT-2's average perplexity on neutral text
distil_baseline = sum(
    compute_perplexity(t, distil_tokenizer, distil_model) 
    for t in calibration_texts
) / len(calibration_texts)

print(f"GPT-2 baseline:      {gpt2_baseline:.2f}")
print(f"DistilGPT-2 baseline:{distil_baseline:.2f}")

def normalized_score(text, tokenizer, model, baseline):
    raw = compute_perplexity(text, tokenizer, model)
    # Divide by baseline — scores below 1.0 mean model finds it natural
    # Scores above 1.0 mean model finds it unusual
    return raw / baseline

# Test texts
texts = [
    "This paper proposes a novel approach to machine learning",
    "The mitochondria is the powerhouse of the cell",
    "I went to the market to buy some vegetables for dinner",
]

print("\n--- Normalized Results ---")
for text in texts:
    gpt2_norm = normalized_score(text, gpt2_tokenizer, gpt2_model, gpt2_baseline)
    distil_norm = normalized_score(text, distil_tokenizer, distil_model, distil_baseline)
    
    winner = "GPT-2" if gpt2_norm < distil_norm else "DistilGPT-2"
    
    print(f"\nText: '{text}'")
    print(f"  GPT-2 normalized:       {gpt2_norm:.3f}")
    print(f"  DistilGPT-2 normalized: {distil_norm:.3f}")
    print(f"  More likely author: {winner}")