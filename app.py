from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

app = FastAPI()

# Model ve tokenizer'ı yükle
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class InputText(BaseModel):
    text: str

# CBT öneri sözlüğü
cbt_recommendations = {
    "NEGATIVE": ["Zor zamanlar geçici olabilir. Kendine biraz zaman tanı ve küçük şeylerde iyilik ara. Unutma, bu düşünceler senin tüm gerçeğin değil 💙",
                 
                 ],
    "NEUTRAL": ["Hayat bazen durağan olabilir ama bu, gelişim için harika bir fırsat olabilir. Küçük bir değişiklik bile ruh halini etkileyebilir 🌿",
                
                ],
    "POSITIVE": ["Bu enerjiyi sürdürmeye çalış! Şükrettiğin, seni mutlu eden şeyleri not al. Güçlü yönlerine odaklan ✨",
                 
                 ]
}

@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()
    label = model.config.id2label[predicted]
    
    # CBT önerisini çek

    cbt_response = random.choice(cbt_recommendations.get(label, ["Seni dinlemeye hazırım 💌"]))


    return {
        "label": label,
        "confidence": confidence,
        "cbt_response": cbt_response
    }
