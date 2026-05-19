import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

print("Loading models...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

distil_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distil_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
distil_model.eval()

def get_token_surprises(text, tokenizer, model):
    """
    For every token in the text, compute how surprised the model was.
    Returns list of (token_text, surprise_score) pairs.
    
    Surprise score = negative log probability of that token.
    High score = model was very surprised = suspicious token
    Low score = model expected this token = natural token
    """
    inputs = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    
    # logits shape: [1, sequence_length, vocab_size]
    logits = outputs.logits[0]
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    token_surprises = []
    
    # For each position, get the probability of the ACTUAL next token
    # inputs[0][i+1] is the actual token at position i+1
    for i in range(len(inputs[0]) - 1):
        actual_next_token = inputs[0][i + 1].item()
        
        # Probability the model assigned to the actual next token
        prob_of_actual = probs[i][actual_next_token].item()
        
        # Surprise = -log(probability)
        # If probability was 0.9 → surprise = -log(0.9) = 0.1 (low surprise)
        # If probability was 0.01 → surprise = -log(0.01) = 4.6 (high surprise)
        surprise = -np.log(prob_of_actual + 1e-10)
        
        # Get the text of the actual token
        token_text = tokenizer.decode([actual_next_token])
        
        token_surprises.append((token_text, surprise))
    
    return token_surprises

def normalize_surprises(surprises):
    """
    Normalize surprise scores to 0-1 range.
    0 = least suspicious, 1 = most suspicious.
    This makes scores comparable across different texts.
    """
    scores = [s for _, s in surprises]
    min_s = min(scores)
    max_s = max(scores)
    
    if max_s == min_s:
        return [(t, 0.5) for t, _ in surprises]
    
    normalized = [
        (token, (score - min_s) / (max_s - min_s))
        for token, score in surprises
    ]
    return normalized

def analyze_tokens(text):
    """
    Analyze text token by token and show suspicion scores.
    """
    print(f"\nText: '{text}'")
    print("-" * 50)
    
    # Get surprises from both models
    gpt2_surprises   = get_token_surprises(text, gpt2_tokenizer, gpt2_model)
    distil_surprises = get_token_surprises(text, distil_tokenizer, distil_model)
    
    # Normalize both
    gpt2_norm   = normalize_surprises(gpt2_surprises)
    distil_norm = normalize_surprises(distil_surprises)
    
    # Print token by token
    print(f"{'Token':<15} {'GPT-2 suspicion':>17} {'DistilGPT-2':>12}")
    print("-" * 50)
    
    for i in range(min(len(gpt2_norm), len(distil_norm))):
        token      = gpt2_norm[i][0]
        gpt2_score = gpt2_norm[i][1]
        dist_score = distil_norm[i][1]
        
        # Visual bar — more stars = more suspicious
        gpt2_bar = "█" * int(gpt2_score * 10)
        
        print(f"{token:<15} {gpt2_score:>6.3f} {gpt2_bar:<10} {dist_score:>6.3f}")

# Test
texts = [
    "This paper proposes a novel approach to machine learning",
    "yo bro what r u doing later lmk",
]

for text in texts:
    analyze_tokens(text)