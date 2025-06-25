import os
import torch
import joblib
import streamlit as st
from transformers import DistilBertModel, DistilBertTokenizerFast
import torch.nn as nn

# ─── Setup ────────────────────────────────────────────────────────────────
model_dir = r"U:\N\save"  # Adjust this if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definition ─────────────────────────────────────────────────────
class ChatModerationModel(nn.Module):
    def __init__(self, num_labels, num_statuses):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for layer in self.bert.transformer.layer[:3]:
            for param in layer.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.6)
        self.label_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_labels)
        )
        self.status_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_statuses)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.label_head(pooled), self.status_head(pooled)

# ─── Load Encoders & Tokenizer ────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
label_enc = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
status_enc = joblib.load(os.path.join(model_dir, "status_encoder.pkl"))

status_map = {0: 'accepted', 1: 'accepted', 2: 'accepted',
              3: 'pending',  4: 'pending',
              5: 'blocked',  6: 'blocked', 7: 'blocked'}

# ─── Load Model ──────────────────────────────────────────────────────────
model = ChatModerationModel(
    num_labels=len(label_enc.classes_),
    num_statuses=len(status_enc.classes_)
).to(device)

model.load_state_dict(torch.load(os.path.join(model_dir, "model_state.pt"), map_location=device))
model.eval()

# ─── Prediction Function ─────────────────────────────────────────────────
def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits_label, logits_status = model(**inputs)

    lbl_idx = logits_label.argmax(dim=1).item()
    stt_idx = logits_status.argmax(dim=1).item()

    return {
        "label": label_enc.inverse_transform([lbl_idx])[0],
        "status": status_enc.inverse_transform([stt_idx])[0],
        "decision": status_map[stt_idx]
    }

# ─── Streamlit UI ────────────────────────────────────────────────────────
st.title("🛡️ Chat Moderation Interface")
st.markdown("Enter a message below to test the moderation model:")

text = st.text_area("User Message", height=150)

if st.button("Moderate"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(text)
        st.success(f"**Decision:** {result['decision'].upper()}")
        st.write(f"**Predicted Label:** {result['label']}")
        st.write(f"**Moderation Status:** {result['status']}")
