# ============================================
# ❌ ERROR ANALYSIS - LOKALNO (Koristi spremljene predictions)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================
# 1. PATHOVI (prilagodi prema tvojoj strukturi)
# ============================================

# Tvoj najbolji model (iz slike)
ADAPTER_PATH = "C:/Users/Korisnik/Desktop/Faks/nlp/projekt/PULSE/emotion_classificator/mlruns/864618888908094463/1418e429ab8d4fce9b1b53c083726495/artifacts"  # ili "./mlruns/1418e425ab8d4fce9b1b/artifacts/"

# Dataset
CSV_PATH = "C:/Users/Korisnik/Desktop/Faks/nlp/projekt/archive/final_dataset.csv"  # Tvoj dataset

# MLflow artifacts (ako želiš učitati confusion matrix i report)
MLFLOW_ARTIFACTS = "./mlruns/864618888908094463/1418e429ab8d4fce9b1b53c083726495/artifacts"

print("=" * 70)
print("🔄 SETUP")
print("=" * 70)
print(f"Adapter path: {ADAPTER_PATH}")
print(f"Dataset path: {CSV_PATH}")

# ============================================
# 2. UČITAJ MODEL I TOKENIZER
# ============================================

print("\n🔄 Učitavam model i tokenizer...")

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    id2label={0: 'joy', 1: 'sadness', 2: 'anger', 3: 'love', 4: 'surprise', 5: 'neutral'},
    label2id={'joy': 0, 'sadness': 1, 'anger': 2, 'love': 3, 'surprise': 4, 'neutral': 5}
)

# Učitaj LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"✅ Model učitan na: {device}")

# ============================================
# 3. UČITAJ I PODIJELI DATASET (ISTI SPLIT KAO U TRENINGU)
# ============================================

print("\n🔄 Učitavam dataset...")

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['text', 'emotion']).copy()

# Label mapping (isti kao u treningu)
LABEL_MAP = {
    'happiness': 'joy', 'enthusiasm': 'joy', 'fun': 'joy', 'relief': 'joy',
    'love': 'love',
    'sadness': 'sadness', 'empty': 'sadness',
    'anger': 'anger', 'hate': 'anger',
    'neutral': 'neutral',
    'surprise': 'surprise'
}

df['emotion'] = df['emotion'].map(lambda x: LABEL_MAP.get(x, x))

# 90-10 split (ISTI random_state=42 kao u treningu!)
train_df, valid_df = train_test_split(
    df, 
    test_size=0.1,  # 10% za validation
    stratify=df['emotion'], 
    random_state=42  # VAŽNO: Isti seed kao u treningu!
)

valid_df = valid_df.reset_index(drop=True)

print(f"✅ Dataset loaded:")
print(f"   Train: {len(train_df)} primjera")
print(f"   Valid: {len(valid_df)} primjera")

# ============================================
# 4. FUNKCIJA ZA PREDIKCIJU
# ============================================

def predict_with_confidence(text, model, tokenizer, device):
    """
    Returns: (predicted_label, confidence, all_probs)
    """
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    pred_idx = np.argmax(probs)
    pred_label = model.config.id2label[pred_idx]
    confidence = probs[pred_idx]
    
    return pred_label, confidence, probs

# ============================================
# 5. GENERIRAJ PREDIKCIJE ZA VALIDATION SET
# ============================================

print("\n🔄 Generiram predikcije...")

predictions = []
confidences = []

for text in tqdm(valid_df['text'], desc="Predicting"):
    pred_label, conf, _ = predict_with_confidence(text, model, tokenizer, device)
    predictions.append(pred_label)
    confidences.append(conf)

valid_df['pred_label'] = predictions
valid_df['confidence'] = confidences

print("✅ Predikcije gotove!")

# Accuracy check
accuracy = (valid_df['emotion'] == valid_df['pred_label']).mean()
print(f"✅ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================
# 6. ANALIZA GREŠAKA
# ============================================

print("\n" + "=" * 70)
print("❌ ANALIZA GREŠAKA")
print("=" * 70)

# Odvoji točne i netočne predikcije
correct_df = valid_df[valid_df['emotion'] == valid_df['pred_label']].copy()
errors_df = valid_df[valid_df['emotion'] != valid_df['pred_label']].copy()

print(f"\n✅ Točnih predikcija: {len(correct_df)} ({len(correct_df)/len(valid_df)*100:.2f}%)")
print(f"❌ Netočnih predikcija: {len(errors_df)} ({len(errors_df)/len(valid_df)*100:.2f}%)")

# Kreiraj error type
errors_df['error_type'] = errors_df['emotion'] + ' → ' + errors_df['pred_label']

# ============================================
# 7. NAJČEŠĆE GREŠKE
# ============================================

print("\n" + "=" * 70)
print("🔝 TOP 10 NAJČEŠĆIH GREŠAKA")
print("=" * 70)

top_errors = errors_df['error_type'].value_counts().head(10)

for i, (error_type, count) in enumerate(top_errors.items(), 1):
    percentage = (count / len(errors_df)) * 100
    print(f"{i:2}. {error_type:25} {count:4} greške ({percentage:5.2f}% svih grešaka)")

# Grafikon
plt.figure(figsize=(12, 6))
top_errors.plot(kind='barh', color='coral')
plt.title('Top 10 najčešćih grešaka modela', fontsize=14, fontweight='bold')
plt.xlabel('Broj grešaka', fontsize=12)
plt.ylabel('Tip greške (Stvarna → Predviđena)', fontsize=12)
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Spremljeno: error_analysis.png")

# ============================================
# 8. GREŠKE PO KLASAMA
# ============================================

print("\n" + "=" * 70)
print("📊 GREŠKE PO KLASAMA")
print("=" * 70)

error_by_class = errors_df['emotion'].value_counts()
total_by_class = valid_df['emotion'].value_counts()

error_rate_df = pd.DataFrame({
    'Ukupno': total_by_class,
    'Greške': error_by_class,
    'Error Rate (%)': (error_by_class / total_by_class * 100).round(2)
}).fillna(0).sort_values('Error Rate (%)', ascending=False)

print(error_rate_df)

# ============================================
# 9. PRIMJERI LOŠIH PREDIKCIJA
# ============================================

print("\n" + "=" * 70)
print("📝 PRIMJERI NAJGORIH PREDIKCIJA (najniži confidence)")
print("=" * 70)

# Top 10 najgorih
worst_errors = errors_df.nsmallest(10, 'confidence')

for i, (idx, row) in enumerate(worst_errors.iterrows(), 1):
    text = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
    
    print(f"\n{'='*70}")
    print(f"🔴 GREŠKA #{i}")
    print(f"{'='*70}")
    print(f"📄 Text: \"{text}\"")
    print(f"✅ Stvarna: {row['emotion'].upper()}")
    print(f"❌ Predviđena: {row['pred_label'].upper()}")
    print(f"📊 Confidence: {row['confidence']:.2%}")

# ============================================
# 10. PRIMJERI ZA TOP 5 TIPOVA GREŠAKA
# ============================================

print("\n" + "=" * 70)
print("📝 PRIMJERI ZA TOP 5 TIPOVA GREŠAKA")
print("=" * 70)

for error_type in top_errors.head(5).index:
    print(f"\n{'='*70}")
    print(f"🔸 TIP GREŠKE: {error_type}")
    print(f"{'='*70}")
    
    examples = errors_df[errors_df['error_type'] == error_type].head(3)
    
    for j, (idx, row) in enumerate(examples.iterrows(), 1):
        text = row['text'][:120] + "..." if len(row['text']) > 120 else row['text']
        print(f"\n  {j}. \"{text}\"")
        print(f"     Confidence: {row['confidence']:.2%}")

# ============================================
# 11. CONFIDENCE DISTRIBUCIJA
# ============================================

print("\n" + "=" * 70)
print("📊 DISTRIBUCIJA CONFIDENCE")
print("=" * 70)

print(f"Prosječni confidence (točne predikcije): {correct_df['confidence'].mean():.2%}")
print(f"Prosječni confidence (netočne predikcije): {errors_df['confidence'].mean():.2%}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(correct_df['confidence'], bins=50, alpha=0.7, color='green', label='Točne')
plt.hist(errors_df['confidence'], bins=50, alpha=0.7, color='red', label='Netočne')
plt.xlabel('Confidence', fontsize=12)
plt.ylabel('Frekvencija', fontsize=12)
plt.title('Distribucija confidence-a', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([correct_df['confidence'], errors_df['confidence']], labels=['Točne', 'Netočne'])
plt.ylabel('Confidence', fontsize=12)
plt.title('Boxplot confidence-a', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Spremljeno: confidence_distribution.png")

# ============================================
# 12. IZVOZ REZULTATA
# ============================================

print("\n" + "=" * 70)
print("💾 SPREMANJE REZULTATA")
print("=" * 70)

# Spremi sve greške
errors_df[['text', 'emotion', 'pred_label', 'confidence', 'error_type']].to_csv(
    'all_errors.csv', 
    index=False
)

# Spremi top 10 najgorih
worst_errors[['text', 'emotion', 'pred_label', 'confidence']].to_csv(
    'worst_10_errors.csv',
    index=False
)

# Spremi error rate po klasama
error_rate_df.to_csv('error_rate_by_class.csv')

print("✅ Spremljeno:")
print("  - all_errors.csv (sve greške)")
print("  - worst_10_errors.csv (10 najgorih)")
print("  - error_rate_by_class.csv")
print("  - error_analysis.png")
print("  - confidence_distribution.png")

# ============================================
# 13. SAŽETAK ZA DOKUMENTACIJU
# ============================================

print("\n" + "=" * 70)
print("📄 SAŽETAK ZA DOKUMENTACIJU")
print("=" * 70)

najcesca_greska = top_errors.index[0]
najcesca_count = top_errors.iloc[0]
najcesca_pct = (najcesca_count / len(errors_df)) * 100

najteza_klasa = error_rate_df.index[0]
najteza_error_rate = error_rate_df.iloc[0]['Error Rate (%)']

print(f"""
**Analiza grešaka na validation setu:**

Model je postigao **{accuracy*100:.2f}% accuracy** na validation setu od {len(valid_df)} primjera.

**Najčešća greška:**
{najcesca_greska} - {najcesca_count} slučajeva ({najcesca_pct:.1f}% svih grešaka)

**Top 5 najčešćih grešaka:**
{chr(10).join([f"{i}. {et} - {cnt} slučajeva ({cnt/len(errors_df)*100:.1f}%)" for i, (et, cnt) in enumerate(top_errors.head(5).items(), 1)])}

**Najteža klasa:**
{najteza_klasa} s {najteza_error_rate:.1f}% error rate

**Prosječni confidence:**
- Točne predikcije: {correct_df['confidence'].mean():.2%}
- Netočne predikcije: {errors_df['confidence'].mean():.2%}

**Zaključak:**
Model ima najveći problem s razlikovanjem **{najcesca_greska.split(' → ')[0]}** od **{najcesca_greska.split(' → ')[1]}**, 
što ukazuje na semantičku sličnost ovih emocija. Razlika u prosječnom confidence-u između
točnih ({correct_df['confidence'].mean():.2%}) i netočnih ({errors_df['confidence'].mean():.2%}) predikcija ukazuje da model
{"dobro" if (correct_df['confidence'].mean() - errors_df['confidence'].mean()) > 0.15 else "slabo"} razlikuje nesigurne slučajeve.
""")

print("\n" + "=" * 70)
print("✅ GOTOVO! Svi rezultati spremljeni.")
print("=" * 70)
