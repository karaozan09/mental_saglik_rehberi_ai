from transformers import pipeline
import os

model_path = "results\checkpoint-400"  # Using the latest checkpoint

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
        
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

    # Test cümleleri
    test_sentences = [
        "Bugün moralim çok bozuk.",
        "Bugün tuhaf hissediyorum.",
        "sınavım açıklandı, beklediğim gibi değildi!",
        "Bugün insan görmek istemiyorum",
        "Hava çok hoş",
        "Hava çok sıcak",
        "birsürü ödevim var",
        "Bugün sevgilim geldi",
        "Stresli hissediyorum",
        "Bugün çok yoruldum",
        "Nerede kaldı ya"
    ]

    print("Model Tahminleri:")
    print("-" * 50)
    for sentence in test_sentences:
        result = classifier(sentence)
        print(f"Cümle: {sentence}")
        print(f"Etiket: {result[0]['label']}")
        print(f"Güven Skoru: {result[0]['score']:.4f}")
        print("-" * 50)
    
except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")
