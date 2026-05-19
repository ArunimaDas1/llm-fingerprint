# LLM Authorship Fingerprinting & Attribution System

A research-grade tool that analyzes any piece of text and determines 
which Large Language Model (LLM) likely wrote it, using statistical 
analysis of token probability distributions.

## What it does
- Input any piece of text
- Outputs which LLM likely wrote it + confidence score
- Shows a token-level heatmap of "suspicion" — which words gave it away
- Uses perplexity scoring, entropy analysis, and normalized attribution

##  The Math Behind It

### Perplexity
Measures how "surprised" a model is by a piece of text. If GPT-2 
assigns high probability to most tokens in a text, it has low 
perplexity on that text — meaning it likely wrote it.

### Entropy
Measures how uncertain a model is at each token position. 
AI-generated text tends to have lower entropy — models always 
pick the "safe" high-probability word.

### Normalization
Raw perplexity scores are not comparable across models since each 
model has a different confidence baseline. We normalize by dividing 
each score by the model's average perplexity on neutral calibration 
text.

### Attribution
We use softmax on negated normalized scores to convert model scores 
into a probability distribution over candidate authors. Lower 
perplexity maps to higher confidence.

## Tech Stack
- Python 3.13
- HuggingFace Transformers
- PyTorch
- Streamlit

##  Models Used
- GPT-2 (117M parameters) — OpenAI
- DistilGPT-2 (82M parameters) — HuggingFace

## 🚀 Run Locally

```bash
git clone https://github.com/ArunimaDas1/llm-fingerprint.git
cd llm-fingerprint
python -m venv venv
venv\Scripts\activate
pip install transformers torch numpy streamlit
streamlit run app.py
```

##  Results
The system shows highest confidence when comparing 
architecturally distinct models. Current implementation 
uses GPT-2 and DistilGPT-2 for lightweight local inference.

##  Research Foundation
This project is inspired by:
- Kirchenbauer et al. (2023) — "A Watermark for Large Language Models"
- The statistical foundations of perplexity-based AI detection

## 🔮 Future Work
- Add GPT-Neo (EleutherAI) for more distinct model fingerprints
- Train a classifier on extracted features for higher accuracy
- Deploy as a public web app
