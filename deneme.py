import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# 1. Excel dosyasını oku
df_raw = pd.read_excel("mental_saglik_verisetim2.xlsx")

# 2. Uzun formata çevir (Her satır: bir cümle ve bir etiket)
# Örn: sütunlar ["very_negative", "negative", "neutral", "positive"]
df_long = pd.melt(df_raw, var_name='label', value_name='text')
df_long.dropna(inplace=True)  # Boş hücreleri kaldır

# 3. Etiketleri sayısallaştır
label2id = {label: i for i, label in enumerate(df_long['label'].unique())}
id2label = {i: label for label, i in label2id.items()}
df_long['label_id'] = df_long['label'].map(label2id)

# 4. Eğitim ve doğrulama setlerini ayır
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_long['text'].tolist(),
    df_long['label_id'].tolist(),
    test_size=0.2,
    random_state=42
)

# 5. Tokenizer hazırla
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# 6. Torch dataset sınıfı
class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MentalHealthDataset(train_encodings, train_labels)
val_dataset = MentalHealthDataset(val_encodings, val_labels)

# 7. Modeli yükle
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 8. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# 10. Eğitimi başlat
trainer.train()
