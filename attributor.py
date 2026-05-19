import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LOAD MODELS ----

print("Loading models...")
models = {
    "GPT-2": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model": GPT2LMHeadModel.from_pretrained("gpt2"),
    },
    "DistilGPT-2": {
        "tokenizer": AutoTokenizer.from_pretrained("distilgpt2"),
        "model": AutoModelForCausalLM.from_pretrained("distilgpt2"),
    },
}

for name, m in models.items():
    m["model"].eval()
    print(f"  {name} ready")

# ---- CALIBRATION ----
# Find each model's baseline perplexity on neutral text
# So we can normalize later

calibration_texts = [
    "The weather today is sunny and warm",
    "She walked to the store to buy milk",
    "The dog ran across the green field",
    "He opened the book and started reading",
    "They sat together and had dinner",
]

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    return torch.exp(outputs.loss).item()

def compute_entropy(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    logits = outputs.logits[0]
    probs = F.softmax(logits, dim=-1)
    # Average entropy across all token positions
    entropies = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
    return entropies.mean().item()

print("\nCalibrating...")
for name, m in models.items():
    baseline = np.mean([
        compute_perplexity(t, m["tokenizer"], m["model"])
        for t in calibration_texts
    ])
    m["baseline"] = baseline
    print(f"  {name} baseline: {baseline:.2f}")

# ---- ATTRIBUTION ----

def attribute(text):
    """
    Given a piece of text, return which model most likely wrote it
    and a confidence score.
    """
    scores = {}

    for name, m in models.items():
        # Normalized perplexity — lower means model finds text more natural
        raw_perp = compute_perplexity(text, m["tokenizer"], m["model"])
        norm_perp = raw_perp / m["baseline"]

        # Entropy — lower means model is more confident about this text
        entropy = compute_entropy(text, m["tokenizer"], m["model"])

        # Combined score — we weight perplexity more since it's our
        # strongest signal. Lower combined score = more likely author.
        # 70% weight on perplexity, 30% on entropy
        combined = 0.7 * norm_perp + 0.3 * (entropy / 16)
        # We divide entropy by 16 to bring it to same scale as norm_perp
        # (entropy is in bits, typically 0-16 for this vocab size)

        scores[name] = {
            "perplexity": raw_perp,
            "norm_perplexity": norm_perp,
            "entropy": entropy,
            "combined": combined,
        }

    # Convert combined scores to confidence percentages
    # Lower combined score should get HIGHER confidence
    # So we negate before softmax
    model_names = list(scores.keys())
    combined_scores = np.array([scores[n]["combined"] for n in model_names])

    # Softmax on negated scores — lower score becomes higher probability
    negated = -combined_scores
    exp_scores = np.exp(negated - np.max(negated))  # subtract max for stability
    confidences = exp_scores / exp_scores.sum() * 100

    # Build result
    result = {}
    for i, name in enumerate(model_names):
        result[name] = {
            **scores[name],
            "confidence": confidences[i],
        }

    # Sort by confidence
    result = dict(sorted(result.items(),
                  key=lambda x: x[1]["confidence"], reverse=True))
    return result

# ---- TEST ----

test_texts = [
    "This paper proposes a novel approach to machine learning",
    "I went to the market to buy some vegetables for dinner",
    "The mitochondria is the powerhouse of the cell",
    "yo bro what r u doing later lmk",
]

print("\n" + "="*55)
for text in test_texts:
    print(f"\nText: '{text}'")
    results = attribute(text)
    for model_name, data in results.items():
        print(f"  {model_name}: {data['confidence']:.1f}% confident")
        print(f"    perplexity={data['perplexity']:.1f}, "
              f"entropy={data['entropy']:.2f}")
    winner = list(results.keys())[0]
    confidence = list(results.values())[0]['confidence']
    print(f"  → Verdict: {winner} ({confidence:.1f}% confidence)")
    print()