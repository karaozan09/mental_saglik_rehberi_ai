from transformers import pipeline
import os
import random
import logging
from collections import deque

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
    "Bu duyguyu yaşamanız harika! Bu olumlu deneyimi günlüğünüze kaydetmek ister misiniz? Böylece zor zamanlarda bu anı hatırlayabilirsiniz.",
    "Harika! Bu olumlu duygu durumunun altında yatan düşünce ve davranışlarınızı fark ettiniz mi? Bunları günlük rutininize ekleyebilirsiniz.",
    "Bu olumlu duyguyu sürdürmek için kendinize küçük ödüller verebilirsiniz. Örneğin, sevdiğiniz bir aktiviteyi planlamak gibi.",
    "Bu enerjiyi başkalarıyla paylaşmak da size iyi gelebilir. Belki bir arkadaşınızla bu olumlu deneyimi paylaşabilirsiniz.",
    "Bu olumlu ruh halinizi sürdürmek için mindfulness tekniklerini deneyebilirsiniz. Şu anı tam olarak yaşamak ve kaydetmek önemli."
],
                "NÖTR": [
    "Bu durumu biraz daha detaylı düşünelim. Bu düşüncelerin altında yatan inançlarınız neler olabilir?",
    "Bu durumla ilgili alternatif düşünce yolları var mı? Farklı bir perspektiften bakmayı denediniz mi?",
    "Bu durumun sizi nasıl etkilediğini düşünüyorsunuz? Düşünce-duygu-davranış döngüsünü görebiliyor musunuz?",
    "Bu durumla ilgili kanıtları değerlendirelim. Düşüncelerinizi destekleyen ve desteklemeyen kanıtlar neler?",
    "Bu durumu bir arkadaşınız yaşasaydı, ona nasıl bir tavsiye verirdiniz? Bazen kendimize dışarıdan bakmak faydalı olabilir."
],
                "NEGATİF": {
                   "anlama_ve_destek": [
    "Bu duyguları yaşamanız zor olmalı. Öncelikle kendinize karşı nazik olun. Bu duygular geçici ve normal.",
    "Bu düşüncelerin altında yatan inançlarınızı keşfetmek ister misiniz? Bazen düşüncelerimiz duygularımızı etkiler.",
    "Bu durumla baş etmek için kullandığınız stratejiler var mı? Belki birlikte yeni yöntemler geliştirebiliriz.",
    "Bu duyguların yoğunluğunu 1-10 arasında değerlendirir misiniz? Bu, duygularınızı daha iyi anlamamıza yardımcı olur.",
    "Bu durumla ilgili en kötü senaryo nedir? Ve en iyi senaryo? Gerçekçi bir senaryo düşünürsek ne olur?"
],
                    "nefes_egzersizleri": [
    "Şimdi birlikte bir nefes egzersizi yapalım. Burnunuzdan 4 saniyede nefes alın, 7 saniye tutun ve 8 saniyede verin. Bu, sinir sisteminizi dengelemeye yardımcı olacak.",
    "Gözlerinizi kapatın ve nefesinize odaklanın. Her nefeste vücudunuzun nasıl hissettiğini fark edin. Bu anda kalmanıza yardımcı olacak.",
    "5-4-3-2-1 tekniğini deneyelim: Şu anda gördüğünüz 5 şeyi, duyduğunuz 4 sesi, dokunabildiğiniz 3 şeyi, koklayabildiğiniz 2 şeyi ve tadabildiğiniz 1 şeyi sayın.",
    "Düşüncelerinizi yargılamadan gözlemleyin. Onları bulutlar gibi düşünün - gelip geçiyorlar. Siz onlar değilsiniz.",
    "Şu anda bedeninizde gerginlik hissettiğiniz yerler var mı? O bölgelere odaklanıp nefes alarak gevşemeyi deneyin."
],
                    "aktivite_onerileri": [
    "Kısa bir yürüyüş yapmayı deneyin. Hareket etmek ve temiz hava almak ruh halinizi iyileştirebilir. Yürürken çevrenizdeki güzel şeylere odaklanın.",
    "Sizi rahatlatan bir müzik açın ve 5 dakika boyunca sadece müziğe odaklanın. Düşüncelerinizi müzikle birlikte akışa bırakın.",
    "Günlüğünüze bu duygularınızı yazın. Yazarken kendinizi yargılamayın, sadece gözlemleyin. Bu, duygularınızı daha iyi anlamanıza yardımcı olabilir.",
    "Sevdiğiniz bir içeceği hazırlayın ve onu yavaşça, farkındalıkla için. Her yudumun tadını, sıcaklığını ve kokusunu fark edin.",
    "Basit bir mindfulness egzersizi yapın: Şu anda yaptığınız işe tamamen odaklanın. Düşünceleriniz başka yerlere giderse, nazikçe nefesinize geri dönün."
],
                    "profesyonel_yardim": [
    "Bu duygularla baş etmekte zorlandığınızı görüyorum. Bir uzmandan destek almak, bu süreçte size yardımcı olabilir. BDT terapistleri bu konuda özel eğitimlidir.",
    "Duygularınızın yoğunluğu ve süresi, profesyonel destek almanın faydalı olabileceğini gösteriyor. Bu bir zayıflık değil, kendinize verdiğiniz bir hediyedir.",
    "BDT tekniklerini bir uzmanla birlikte uygulamak, size daha etkili stratejiler geliştirmenizde yardımcı olabilir. Bu süreçte yalnız değilsiniz.",
    "Düşünce-duygu-davranış döngüsünü kırmak için profesyonel destek almak, size yeni perspektifler kazandırabilir. Bu, uzun vadeli iyileşme için önemli bir adım olabilir.",
    "Terapistinizle birlikte çalışarak, bu zor duygularla baş etme becerilerinizi geliştirebilir ve daha sağlıklı düşünce kalıpları oluşturabilirsiniz."
]
                }
            }
            
            self.sohbet_gecmisi = []
            self.son_etiket = None
            self.son_yanit = None
            self.negatif_mesaj_sayisi = 0
            self.terapi_hedefi = None
            
        except Exception as e:
            logger.error(f"BDT sistemi başlatılırken hata oluştu: {str(e)}")
            raise
    
    def terapi_hedefi_belirle(self, metin, etiket):
        """Mevcut duruma göre terapi hedefi belirler"""
        if etiket == "NEGATİF":
            if "kaygı" in metin.lower() or "endişe" in metin.lower():
                return "kaygı yönetimi"
            elif "öfke" in metin.lower() or "sinir" in metin.lower():
                return "öfke kontrolü"
            elif "üzüntü" in metin.lower() or "mutsuz" in metin.lower():
                return "duygu düzenleme"
            elif "stres" in metin.lower() or "gergin" in metin.lower():
                return "stres yönetimi"
        return None
    
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
                guven = min(guven + 0.1, 1.0)
            
            # Terapi hedefi belirleme
            if not self.terapi_hedefi:
                self.terapi_hedefi = self.terapi_hedefi_belirle(metin, etiket)
            
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
                
                # Terapi hedefine göre özel yanıt
                if self.terapi_hedefi:
                    if self.terapi_hedefi == "kaygı yönetimi":
                        yanit += "\n\nKaygıyla baş etmek için nefes egzersizleri çok faydalı olabilir. Hadi birlikte bir egzersiz yapalım: " + random.choice(self.yanitlar["NEGATİF"]["nefes_egzersizleri"])
                    elif self.terapi_hedefi == "öfke kontrolü":
                        yanit += "\n\nÖfke kontrolü için mindfulness teknikleri etkili olabilir. " + random.choice(self.yanitlar["NEGATİF"]["aktivite_onerileri"])
                    elif self.terapi_hedefi == "duygu düzenleme":
                        yanit += "\n\nDuygularınızı düzenlemek için günlük tutmak faydalı olabilir. " + random.choice(self.yanitlar["NEGATİF"]["aktivite_onerileri"])
                    elif self.terapi_hedefi == "stres yönetimi":
                        yanit += "\n\nStres yönetimi için fiziksel aktivite önemlidir. " + random.choice(self.yanitlar["NEGATİF"]["aktivite_onerileri"])
                
                # Profesyonel yardım önerisi
                if self.negatif_mesaj_sayisi >= 3:
                    yanit += "\n\n" + random.choice(self.yanitlar["NEGATİF"]["profesyonel_yardim"])
            else:
                # Pozitif veya nötr yanıtlar için
                yanit = random.choice(self.yanitlar[etiket])
            
            self.son_etiket = etiket
            self.son_yanit = yanit
            
            return {
                'etiket': etiket,
                'guven': guven,
                'yanit': yanit,
                'terapi_hedefi': self.terapi_hedefi
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
