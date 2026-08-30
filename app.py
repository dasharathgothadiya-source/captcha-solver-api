import os
import io
import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import ddddocr

app = Flask(__name__)
CORS(app)  # Allow cross-origin from browser/userscript

# Initialize OCR engines
print("🔄 Loading OCR engines...")
ocr_fast = ddddocr.DdddOcr(show_ad=False)
ocr_beta = ddddocr.DdddOcr(show_ad=False, beta=True)
print("✅ OCR ready!")

# ==================== PREPROCESSING ====================
def preprocess_ultrafast(image):
    image = image.convert('L')
    if image.width < 100:
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.0)

def preprocess_fast(image):
    image = image.convert('L')
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(2.5)

def preprocess_aggressive(image):
    image = image.convert('L')
    image = image.resize((image.width * 3, image.height * 3), Image.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    return ImageOps.autocontrast(image)

def get_confidence(text, attempts, ptime):
    score = 60
    if 4 <= len(text) <= 8: score += 15
    if text.isalnum(): score += 15
    if attempts == 1 and ptime < 0.2: score += 10
    return min(score, 99)

# ==================== SOLVE ENDPOINT ====================
@app.route('/solve', methods=['POST'])
def solve():
    start_time = time.time()

    try:
        data = request.get_json()
        img_data = data.get('image', '')

        # Remove data URI prefix if present
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        image_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(image_bytes))

        result_text = ""
        attempts = 0

        # FAST PATH 1: Ultra-fast
        if image.width >= 150:
            processed = preprocess_ultrafast(image)
            result_text = ocr_fast.classification(processed).strip().replace(' ', '')
            attempts = 1

        # FAST PATH 2: Standard
        if len(result_text) < 3:
            processed = preprocess_fast(image)
            result_text = ocr_fast.classification(processed).strip().replace(' ', '')
            attempts = 2

        # FALLBACK: Aggressive
        if len(result_text) < 3:
            processed = preprocess_aggressive(image)
            result_text = ocr_fast.classification(processed).strip().replace(' ', '')
            attempts = 3

            if len(result_text) < 3:
                result_text = ocr_beta.classification(processed).strip().replace(' ', '')

        processing_time = round(time.time() - start_time, 3)
        confidence = get_confidence(result_text, attempts, processing_time)
        auto_accept = confidence >= 80

        return jsonify({
            'status': 'success',
            'captcha': result_text,
            'confidence': confidence,
            'auto_accept': auto_accept,
            'attempts': attempts,
            'processing_time': processing_time
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'captcha': '',
            'confidence': 0,
            'error': str(e)
        }), 500

# Health check
@app.route('/')
def health():
    return jsonify({'status': 'ok', 'service': 'captcha-solver'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
