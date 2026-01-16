import streamlit as st
import torch
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.tokenization_bart import BartTokenizer

st.set_page_config(page_title="News Summarizer", layout="centered")
st.title("📰 News Summarizer (LoRA Fine-tuned)")

@st.cache_resource
def load_model():
    model = BartForConditionalGeneration.from_pretrained(
        "bart_lora_fast",
        local_files_only=True
    )
    tokenizer = BartTokenizer.from_pretrained(
        "bart_lora_fast",
        local_files_only=True
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

text = st.text_area("Paste news article", height=250)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please paste a news article.")
    else:
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
            num_beams=2,
            early_stopping=True,
                    length_penalty=1.0

        )


        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        print('summary:', summary)
        if not summary:
            st.warning("The model could not generate a meaningful summary for this input.")
        else:
            st.subheader("Summary")
            st.success(summary)
       
