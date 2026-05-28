from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import os
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN", "")

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image using FLUX.1-schnell — free & actively supported"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        width = data.get('width', 1024)
        height = data.get('height', 1024)

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        if not HF_TOKEN:
            return jsonify({'error': 'HuggingFace token not configured.'}), 401

        print(f"Generating image: {prompt[:60]}...")
        print(f"Parameters: {width}x{height}")

        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )

        # FLUX.1-schnell: free, fast, no license gate, actively supported
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=width,
            height=height,
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        print("✓ Image generated successfully!")

        return jsonify({
            'success': True,
            'image': img_base64,
            'message': 'Image generated successfully'
        })

    except Exception as e:
        error_str = str(e)
        print(f"Error: {error_str}")

        if "402" in error_str:
            return jsonify({'error': 'Credits required', 'message': error_str}), 402
        if "401" in error_str or "unauthorized" in error_str.lower():
            return jsonify({'error': 'Invalid HuggingFace token', 'message': 'Check your HF_TOKEN in .env file.'}), 401
        if "403" in error_str:
            return jsonify({'error': 'Access denied', 'message': 'Check model access on HuggingFace.'}), 403
        if "410" in error_str:
            return jsonify({'error': 'Model deprecated', 'message': 'This model is no longer supported.'}), 410
        if "503" in error_str or "loading" in error_str.lower():
            return jsonify({'error': 'Model loading', 'message': 'Please wait 20-30 seconds and try again.'}), 503

        return jsonify({'error': 'Internal server error', 'message': error_str}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'FLUX.1-schnell (free)',
        'provider': 'hf-inference',
        'token_configured': bool(HF_TOKEN),
    })

@app.route('/')
def index():
    return send_from_directory('views', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('views', filename)


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" 🎨 FLUX SCHNELL IMAGE GENERATOR SERVER")
    print("="*70)
    print(f" Server:   http://localhost:5000")
    print(f" Model:    black-forest-labs/FLUX.1-schnell")
    print(f" Provider: hf-inference (FREE)")
    print("="*70)

    if not HF_TOKEN:
        print("\n ⚠️  WARNING: HuggingFace token not configured!")
        print(" 1. Get token: https://huggingface.co/settings/tokens")
        print(" 2. Create .env: HF_TOKEN=your_token_here\n")
    else:
        print(f"\n ✓ Token configured — Ready!\n")

    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)