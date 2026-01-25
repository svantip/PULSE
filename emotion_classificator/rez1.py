import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # Bitno!

# ============================================
# 1. POSTAVKE (Pathovi)
# ============================================
ADAPTER_PATH = "C:/Users/Korisnik/Desktop/Faks/nlp/projekt/PULSE/emotion_classificator/mlruns/864618888908094463/1418e429ab8d4fce9b1b53c083726495/artifacts"
CSV_PATH = "C:/Users/Korisnik/Desktop/Faks/nlp/projekt/archive/final_dataset.csv"
MODEL_NAME = "roberta-base"

print(f"🔄 Učitavam model iz: {ADAPTER_PATH}")

# ============================================
# 2. UČITAJ MODEL
# ============================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=6,
    id2label={0: 'joy', 1: 'sadness', 2: 'anger', 3: 'love', 4: 'surprise', 5: 'neutral'},
    label2id={'joy': 0, 'sadness': 1, 'anger': 2, 'love': 3, 'surprise': 4, 'neutral': 5}
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ============================================
# 3. PRIPREMI PODATKE (Valid set)
# ============================================
print("🔄 Pripremam podatke...")
df = pd.read_csv(CSV_PATH).dropna(subset=['text', 'emotion'])
LABEL_MAP = {'happiness': 'joy', 'enthusiasm': 'joy', 'fun': 'joy', 'relief': 'joy', 'love': 'love', 'sadness': 'sadness', 'empty': 'sadness', 'anger': 'anger', 'hate': 'anger', 'neutral': 'neutral', 'surprise': 'surprise'}
df['emotion'] = df['emotion'].map(lambda x: LABEL_MAP.get(x, x))

_, valid_df = train_test_split(df, test_size=0.1, stratify=df['emotion'], random_state=42)
valid_df = valid_df.reset_index(drop=True)
print(f"✅ Valid set veličina: {len(valid_df)}")

# ============================================
# 4. PREDIKCIJA
# ============================================
print("🔄 Radim predikcije (ovo može potrajati)...")
predictions = []

# Batch processing za brže izvođenje (opcionalno, ovdje simple loop)
for text in tqdm(valid_df['text']):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(model.config.id2label[pred_idx])

valid_df['pred_label'] = predictions
print("✅ Predikcije gotove!")

# ============================================
# 5. MATRICA KONFUZIJE (Tvoj traženi dio)
# ============================================
print("\n📊 Crtam matricu konfuzije...")
labels = ['joy', 'sadness', 'anger', 'love', 'surprise', 'neutral']
cm = confusion_matrix(valid_df['emotion'], valid_df['pred_label'], labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrica Konfuzije')
plt.ylabel('Stvarna Klasa')
plt.xlabel('Predviđena Klasa')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("✅ Spremljeno: confusion_matrix.png")
