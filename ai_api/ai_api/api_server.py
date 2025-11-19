from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        data = request.get_json()
        img_b64 = data['image']
        scale = int(data.get('scale', 4))
        
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        
        w, h = img.size
        new_w = w * scale
        new_h = h * scale
        
        upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        upscaled.save(buffer, format='JPEG', quality=99)
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'upscaled_image': result_b64,
            'size': f"{new_w}x{new_h}"
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)

