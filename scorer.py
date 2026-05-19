import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# ---- LOAD MODELS ----

print("Loading GPT-2...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

print("Loading DistilGPT-2...")
distil_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distil_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
distil_model.eval()

print("Models loaded.\n")

# ---- CORE FUNCTIONS ----

def get_token_distributions(text, tokenizer, model):
    """
    For every token in the text, get the full probability distribution
    over all possible next tokens.
    
    Returns:
    - token_ids: the actual tokens in the text
    - distributions: list of probability arrays, one per token position
    - loss: the average negative log probability (used for perplexity)
    """
    inputs = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    
    # outputs.logits shape: [1, sequence_length, vocab_size]
    # logits are raw scores — we convert to probabilities with softmax
    # Think of logits as "votes" and softmax as converting votes to percentages
    logits = outputs.logits[0]  # remove batch dimension
    probs = F.softmax(logits, dim=-1)  # convert to probabilities
    
    return inputs[0], probs, outputs.loss.item()

def compute_entropy(probs):
    """
    Entropy of a probability distribution.
    High entropy = uncertain = spread out
    Low entropy = certain = one dominant option
    
    Formula: H = -sum(p * log(p))
    We use log base 2 so entropy is measured in bits.
    """
    # Add small number to avoid log(0) which is undefined
    probs = probs + 1e-10
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()

def compute_kl_divergence(probs_p, probs_q):
    """
    KL divergence from distribution P to distribution Q.
    Measures how different Q is from P.
    Low KL = distributions are similar = models agree
    High KL = distributions are different = models disagree
    
    Formula: KL(P||Q) = sum(P * log(P/Q))
    """
    probs_p = probs_p + 1e-10
    probs_q = probs_q + 1e-10
    kl = torch.sum(probs_p * torch.log(probs_p / probs_q))
    return kl.item()

def analyze_text(text):
    """
    Full analysis of a piece of text.
    Returns perplexity, entropy, and KL divergence for both models.
    """
    print(f"Analyzing: '{text}'")
    
    # Get distributions from both models
    tokens_gpt2,   probs_gpt2,   loss_gpt2   = get_token_distributions(
        text, gpt2_tokenizer, gpt2_model
    )
    tokens_distil, probs_distil, loss_distil = get_token_distributions(
        text, distil_tokenizer, distil_model
    )
    
    # Perplexity
    perplexity_gpt2   = np.exp(loss_gpt2)
    perplexity_distil = np.exp(loss_distil)
    
    # Entropy — average across all token positions
    # We skip the last token because it has no "next token" to predict
    entropy_gpt2 = np.mean([
        compute_entropy(probs_gpt2[i]) 
        for i in range(len(probs_gpt2) - 1)
    ])
    entropy_distil = np.mean([
        compute_entropy(probs_distil[i]) 
        for i in range(len(probs_distil) - 1)
    ])
    
    # KL Divergence — how different are the two models at each token?
    # We need both models to use the same vocab size for this
    # GPT-2 and DistilGPT-2 both have vocab size 50257 so we're fine
    min_len = min(len(probs_gpt2), len(probs_distil))
    kl_scores = [
        compute_kl_divergence(probs_gpt2[i], probs_distil[i])
        for i in range(min_len - 1)
    ]
    avg_kl = np.mean(kl_scores)
    
    return {
        "perplexity_gpt2":   perplexity_gpt2,
        "perplexity_distil": perplexity_distil,
        "entropy_gpt2":      entropy_gpt2,
        "entropy_distil":    entropy_distil,
        "avg_kl_divergence": avg_kl,
    }

# ---- TEST IT ----

texts = [
    "This paper proposes a novel approach to machine learning",
    "I went to the market to buy some vegetables for dinner",
    "The mitochondria is the powerhouse of the cell",
]

for text in texts:
    results = analyze_text(text)
    print(f"  Perplexity  — GPT-2: {results['perplexity_gpt2']:.2f} | "
          f"DistilGPT-2: {results['perplexity_distil']:.2f}")
    print(f"  Entropy     — GPT-2: {results['entropy_gpt2']:.2f} | "
          f"DistilGPT-2: {results['entropy_distil']:.2f}")
    print(f"  KL Divergence (avg): {results['avg_kl_divergence']:.4f}")
    print()