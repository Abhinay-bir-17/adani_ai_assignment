import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import PeftModel

st.title("📰 News Summarizer (LoRA Fine-tuned)")

@st.cache_resource
def load_model():
    base = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn"
    )
    model = PeftModel.from_pretrained(base, "./bart_lora_fast")
    tokenizer = BartTokenizer.from_pretrained("./bart_lora_fast")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

text = st.text_area("Paste news article", height=250)

if st.button("Summarize"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=48,
            num_beams=2
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    st.success(summary)
