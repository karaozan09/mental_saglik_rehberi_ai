from flask import Flask, request, jsonify
from flask_cors import CORS
from bdt_yanit_sistemi import BDTYanitSistemi
import traceback
import logging

# Logging ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing için

# BDT sistemini başlat
try:
    bdt_sistemi = BDTYanitSistemi()
    logger.info("BDT sistemi başarıyla başlatıldı")
except Exception as e:
    logger.error(f"BDT sistemi başlatılırken hata oluştu: {str(e)}")
    logger.error(traceback.format_exc())

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        logger.debug(f"Gelen veri: {data}")
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Text alanı gerekli'
            }), 400
        
        text = data['text']
        logger.debug(f"Analiz edilecek metin: {text}")
        
        sonuc = bdt_sistemi.analiz_et(text)
        logger.debug(f"Analiz sonucu: {sonuc}")
        
        return jsonify({
            'success': True,
            'data': {
                'etiket': sonuc['etiket'],
                'guven': sonuc['guven'],
                'yanit': sonuc['yanit']
            }
        })
        
    except Exception as e:
        logger.error(f"Hata oluştu: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # BDT sisteminin çalışıp çalışmadığını kontrol et
        test_text = "Test mesajı"
        bdt_sistemi.analiz_et(test_text)
        return jsonify({
            'status': 'healthy',
            'message': 'BDT API çalışıyor'
        })
    except Exception as e:
        logger.error(f"Health check hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 