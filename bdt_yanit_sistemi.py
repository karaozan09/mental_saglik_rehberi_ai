from transformers import pipeline
import os
import random
import logging

# Logging ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BDTYanitSistemi:
    def __init__(self, model_path="results/checkpoint-400"):
        try:
            # Model yolunu düzelt
            model_path = model_path.replace("\\", "/")
            logger.debug(f"Model yolu: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model yolu {model_path} bulunamadı")
            
            logger.info("Model yükleniyor...")
            self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
            logger.info("Model başarıyla yüklendi")
            
            # BDT yaklaşımına göre yanıtlar
            self.yanitlar = {
                "POZİTİF": [
                    "Bu olumlu duyguları yaşamanız harika! Bu anı değerlendirmek ve olumlu deneyimlerinizi günlüğünüze kaydetmek ister misiniz?",
                    "Harika bir ruh halindesiniz! Bu olumlu duyguları daha sık yaşamak için neler yapabileceğinizi düşündünüz mü?",
                    "Bu olumlu duygu durumunuzu sürdürmek için kendinize nasıl bir ödül verebilirsiniz?",
                    "Bu olumlu duyguyu tetikleyen şey neydi? Bu deneyimi daha sık yaşamak için neler yapabilirsiniz?",
                    "Bu olumlu duygu durumunuzun hayatınızın diğer alanlarına etkisi nasıl olabilir?",
                    "Bu olumlu deneyimi nasıl değerlendiriyorsunuz? Bu duyguyu daha sık yaşamak için neler yapabilirsiniz?",
                    "Bu olumlu duygu durumunuzu sürdürmek için kendinize nasıl bir ödül verebilirsiniz?",
                    "Bu olumlu duyguları yaşamanız çok güzel! Bu anı değerlendirmek ve olumlu deneyimlerinizi günlüğünüze kaydetmek ister misiniz?"
                ],
                "NÖTR": [
                    "Bu durumu daha detaylı değerlendirmek ister misiniz? Şu anki düşünceleriniz neler?",
                    "Bu durumla ilgili alternatif bakış açıları düşünmek ister misiniz?",
                    "Şu anki durumunuzu değiştirmek istediğiniz bir yön var mı?",
                    "Bu durumun sizin için anlamı nedir?",
                    "Bu durumla ilgili daha detaylı düşünmek ister misiniz?",
                    "Bu durumun hayatınızın diğer alanlarına etkisi nasıl?",
                    "Bu durumu daha olumlu bir şekilde değerlendirmek için neler yapabilirsiniz?",
                    "Bu durumla ilgili değiştirmek istediğiniz bir şey var mı?"
                ],
                "NEGATİF": {
                    "anlama_ve_destek": [
                        "Bu duyguyu yaşadığınız için üzgünüm. Sizi anlıyorum ve buradayım.",
                        "Bu duyguları yaşamanız normal. Sizi dinliyorum ve destekliyorum.",
                        "Bu duyguyu yaşadığınızı duyduğuma üzüldüm. Sizinle konuşmak isterim.",
                        "Bu duygularla başa çıkmak zor olabilir. Sizi anlıyorum ve yanınızdayım.",
                        "Bu duyguyu yaşadığınızı duyduğuma üzüldüm. Sizinle konuşmak ve size destek olmak isterim."
                    ],
                    "nefes_egzersizleri": [
                        "Şu an kendinize iyi gelmek için 4-7-8 nefes tekniğini deneyebilirsiniz: 4 saniye nefes alın, 7 saniye tutun, 8 saniyede verin.",
                        "Nefes egzersizleri yapmak ister misiniz? Bu, sakinleşmenize yardımcı olabilir.",
                        "Şu an kendinize iyi gelmek için derin nefes alabilirsiniz. 4 saniye nefes alın, 4 saniye tutun ve 4 saniyede verin.",
                        "Nefes egzersizleri yapmak ister misiniz? Bu, duygularınızı düzenlemenize yardımcı olabilir."
                    ],
                    "aktivite_onerileri": [
                        "Kısa bir yürüyüşe çıkmak size iyi gelebilir. Doğada vakit geçirmek ruh halinizi olumlu etkileyebilir.",
                        "Sevdiğiniz bir müziği dinlemek ve dans etmek endorfin seviyenizi yükseltebilir.",
                        "Size iyi gelen bir aktiviteye zaman ayırmak ister misiniz? Örneğin resim yapmak, yazı yazmak veya meditasyon yapmak.",
                        "Kısa bir yürüyüşe çıkmak veya hafif egzersiz yapmak size iyi gelebilir."
                    ],
                    "profesyonel_yardim": [
                        "Bu duygularla başa çıkmakta zorlandığınızı görüyorum. Bir uzmandan destek almak size yardımcı olabilir.",
                        "Bu duygular günlük hayatınızı etkiliyorsa, profesyonel destek almak önemli olabilir.",
                        "Bu duygularla başa çıkmak için yalnız olmadığınızı bilin. Bir uzmandan destek almak size yardımcı olabilir.",
                        "Bu duygularla başa çıkmakta zorlandığınızı görüyorum. Bir uzmanla görüşmek ister misiniz?"
                    ]
                }
            }
            
            self.sohbet_gecmisi = []
            self.son_etiket = None
            self.son_yanit = None
            self.negatif_mesaj_sayisi = 0
            
        except Exception as e:
            logger.error(f"BDT sistemi başlatılırken hata oluştu: {str(e)}")
            raise
    
    def analiz_et(self, metin):
        try:
            logger.debug(f"Metin analiz ediliyor: {metin}")
            sonuc = self.classifier(metin)[0]
            etiket = sonuc['label']
            guven = sonuc['score']
            logger.debug(f"Analiz sonucu: etiket={etiket}, guven={guven}")
            
            # Etiket eşleştirme
            if etiket == "ÇOK NEGATİF":
                etiket = "NEGATİF"
                # Çok negatif durumlar için daha güçlü bir yanıt oluştur
                guven = min(guven + 0.1, 1.0)  # Güven skorunu biraz artır
            
            # Sohbet geçmişini güncelle
            self.sohbet_gecmisi.append({
                'metin': metin,
                'etiket': etiket,
                'guven': guven
            })
            
            # Negatif mesaj sayısını güncelle
            if etiket == "NEGATİF":
                self.negatif_mesaj_sayisi += 1
            else:
                self.negatif_mesaj_sayisi = 0
            
            # Yanıt oluşturma stratejisi
            if etiket == "NEGATİF":
                # Anlama ve destek mesajı
                yanit = random.choice(self.yanitlar["NEGATİF"]["anlama_ve_destek"])
                
                # Ek öneriler
                if self.negatif_mesaj_sayisi >= 3:
                    # 3 veya daha fazla negatif mesaj varsa profesyonel yardım öner
                    yanit += "\n\n" + random.choice(self.yanitlar["NEGATİF"]["profesyonel_yardim"])
                else:
                    # Nefes egzersizi veya aktivite önerisi
                    if random.random() < 0.5:
                        yanit += "\n\n" + random.choice(self.yanitlar["NEGATİF"]["nefes_egzersizleri"])
                    else:
                        yanit += "\n\n" + random.choice(self.yanitlar["NEGATİF"]["aktivite_onerileri"])
            else:
                # Pozitif veya nötr yanıtlar için
                yanit = random.choice(self.yanitlar[etiket])
            
            self.son_etiket = etiket
            self.son_yanit = yanit
            
            return {
                'etiket': etiket,
                'guven': guven,
                'yanit': yanit
            }
            
        except Exception as e:
            logger.error(f"Metin analiz edilirken hata oluştu: {str(e)}")
            raise

def main():
    try:
        sistem = BDTYanitSistemi()
        print("\nMerhaba! Ben BDT tabanlı bir duygu analizi ve destek sistemiyim.")
        print("Duygu durumunuzu ve düşüncelerinizi benimle paylaşabilirsiniz.")
        
        while True:
            print("\nSizin için buradayım. Nasıl hissediyorsunuz?")
            metin = input("> ")
            
            if metin.lower() in ['q', 'quit', 'exit', 'çıkış', 'çık', 'bitir']:
                print("\nGörüşmek üzere! Kendinize iyi bakın.")
                break
                
            sonuc = sistem.analiz_et(metin)
            print(f"\n{sonuc['yanit']}")
            
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 