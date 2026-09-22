import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.ui import base_layout

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Text Summarization using T5",
    page_icon="📝",
    layout="wide"
)

# Apply custom UI styling
base_layout()

# ---------------------------------------------------------------------------
# Model and Tokenizer Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading T5 Summarization Model...")
def load_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dir = "./saved_summary_model"
    
    # Check if model weights exist
    weights_exist = any(
        os.path.exists(os.path.join(model_dir, f))
        for f in ["model.safetensors", "pytorch_model.bin"]
    )
    
    if not weights_exist and not os.path.exists(model_dir):
        return None, None, device

    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model from '{model_dir}': {e}")
        return None, None, device

model, tokenizer, device = load_model()

# ---------------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------------
st.markdown("<h2 style='text-align:center;'>Text Summarization</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#555;'>Fine-tuned Hugging Face T5 on SAMSum Dialogue Dataset</h4>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------------------------------
def clean_data(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"\r\n", " ", text)       # line breaks
    text = re.sub(r"<.*?>", " ", text)      # HTML tags
    text = re.sub(r"\s+", " ", text)        # multiple spaces
    text = text.strip().lower()
    return text

# ---------------------------------------------------------------------------
# User Input Section
# ---------------------------------------------------------------------------
if model is None or tokenizer is None:
    st.warning(
        "Model weights not found in `./saved_summary_model/`.\n\n"
        "Please train the model by running `notebook.ipynb` or place `model.safetensors` in the `saved_summary_model` directory."
    )

text_input = st.text_area(
    "Enter dialogue or text to summarize:",
    height=250,
    max_chars=5000,
    placeholder="Example:\nAmanda: I baked cookies. Do you want some?\nJerry: Sure! I'll be over in 10 minutes.\nAmanda: Great, see you soon!"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    summarize_btn = st.button("Summarize", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Inference & Summary Generation
# ---------------------------------------------------------------------------
if summarize_btn and text_input:
    if model is None or tokenizer is None:
        st.error("Cannot summarize: Model is not loaded.")
    else:
        with st.spinner("Generating summary with T5..."):
            cleaned_text = clean_data(text_input)
            inputs = tokenizer(
                cleaned_text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                targets = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(targets[0], skip_special_tokens=True)

        st.markdown("<h3 style='text-align:center;'>Summary</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background-color:#1e1e1e;
                border:1px solid #444;
                border-radius:10px;
                padding:20px;
                margin-top:10px;
                font-size:1.1rem;
                line-height:1.6;
                color:#f5f5f5;
            ">
                {summary}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px; border:0.5px solid #bbb;">
    <div style="text-align:center; color:#555; font-size:0.9rem; padding:10px;">
        Built with ❤️ by Jatin Kumar using Hugging Face Transformers, PyTorch & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)