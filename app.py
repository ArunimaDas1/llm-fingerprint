import streamlit as st
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="LLM Fingerprint Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LLM Authorship Fingerprinting")
st.markdown("Paste any text below to find out which AI model likely wrote it.")

# ---- LOAD MODELS ----
# @st.cache_resource means: load the models once and reuse them
# Without this, models reload every time you click a button — very slow
@st.cache_resource
def load_models():
    st.write("Loading models for the first time — this takes ~30 seconds...")
    
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_mod = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_mod.eval()
    
    distil_tok = AutoTokenizer.from_pretrained("distilgpt2")
    distil_mod = AutoModelForCausalLM.from_pretrained("distilgpt2")
    distil_mod.eval()
    
    return gpt2_tok, gpt2_mod, distil_tok, distil_mod

gpt2_tokenizer, gpt2_model, distil_tokenizer, distil_model = load_models()

# ---- CORE FUNCTIONS ----

def get_token_surprises(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    if len(inputs[0]) < 2:
        return []
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    logits = outputs.logits[0]
    probs = F.softmax(logits, dim=-1)
    
    token_surprises = []
    for i in range(len(inputs[0]) - 1):
        actual_next = inputs[0][i + 1].item()
        prob = probs[i][actual_next].item()
        surprise = -np.log(prob + 1e-10)
        token_text = tokenizer.decode([actual_next])
        token_surprises.append((token_text, surprise))
    
    return token_surprises

def compute_perplexity(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    return torch.exp(outputs.loss).item()

def surprise_to_color(score):
    """
    Convert a 0-1 suspicion score to a color.
    0 = green (natural), 1 = red (suspicious)
    Returns an RGB color string.
    """
    # Interpolate between green (0,200,0) and red (220,0,0)
    r = int(220 * score)
    g = int(200 * (1 - score))
    b = 0
    return f"rgb({r},{g},{b})"

def normalize(surprises):
    scores = [s for _, s in surprises]
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(t, 0.5) for t, _ in surprises]
    return [(t, (s - min_s)/(max_s - min_s)) for t, s in surprises]

def get_confidence(text):
    """
    Returns confidence scores for each model.
    """
    calibration = [
        "The weather today is sunny and warm",
        "She walked to the store to buy milk",
        "The dog ran across the green field",
    ]
    
    gpt2_baseline = np.mean([
        compute_perplexity(t, gpt2_tokenizer, gpt2_model) 
        for t in calibration
    ])
    distil_baseline = np.mean([
        compute_perplexity(t, distil_tokenizer, distil_model) 
        for t in calibration
    ])
    
    gpt2_perp   = compute_perplexity(text, gpt2_tokenizer, gpt2_model)
    distil_perp = compute_perplexity(text, distil_tokenizer, distil_model)
    
    gpt2_norm   = gpt2_perp / gpt2_baseline
    distil_norm = distil_perp / distil_baseline
    
    # Softmax on negated scores
    scores = np.array([gpt2_norm, distil_norm])
    negated = -scores
    exp_s = np.exp(negated - np.max(negated))
    confidences = exp_s / exp_s.sum() * 100
    
    return {
        "GPT-2": {"confidence": confidences[0], "perplexity": gpt2_perp},
        "DistilGPT-2": {"confidence": confidences[1], "perplexity": distil_perp},
    }

# ---- UI ----

text_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Paste any text here — human written or AI generated..."
)

if st.button("🔍 Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            
            # Get confidence scores
            results = get_confidence(text_input)
            winner = max(results, key=lambda x: results[x]["confidence"])
            win_conf = results[winner]["confidence"]
            
            # ---- VERDICT ----
            st.markdown("---")
            st.subheader("📊 Verdict")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Likely Author", winner)
            with col2:
                st.metric("Confidence", f"{win_conf:.1f}%")
            with col3:
                st.metric("GPT-2 Perplexity", 
                         f"{results['GPT-2']['perplexity']:.1f}")
            
            # Confidence bars
            st.markdown("#### Model Confidence")
            for model_name, data in results.items():
                st.progress(
                    data["confidence"] / 100,
                    text=f"{model_name}: {data['confidence']:.1f}%"
                )
            
            # ---- HEATMAP ----
            st.markdown("---")
            st.subheader("🌡️ Token Suspicion Heatmap")
            st.markdown("**Green** = model expected this word · "
                       "**Red** = model was surprised")
            
            # Get token surprises from GPT-2
            surprises = get_token_surprises(
                text_input, gpt2_tokenizer, gpt2_model
            )
            normalized = normalize(surprises)
            
            # Build colored HTML for each token
            html = '<div style="line-height:2.5; font-size:18px;">'
            for token, score in normalized:
                color = surprise_to_color(score)
                html += (
                    f'<span style="background-color:{color}; '
                    f'color:white; padding:2px 4px; '
                    f'margin:2px; border-radius:4px;">'
                    f'{token}</span>'
                )
            html += '</div>'
            
            st.markdown(html, unsafe_allow_html=True)
            
            # ---- EXPLANATION ----
            st.markdown("---")
            st.subheader("📖 How this works")
            st.markdown("""
            - **Perplexity** measures how surprised each model is by the text
            - **Lower perplexity** means the model finds the text natural
            - The model with the lowest normalized perplexity is the likely author
            - **Token colors** show which words surprised GPT-2 the most
            """)