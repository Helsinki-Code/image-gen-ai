import os
import base64
import uuid
import requests
from flask import Flask, request, jsonify, send_from_directory

# Vercel will run this script, so we use its app object.
app = Flask(__name__)

# Vercel provides a temporary directory for file storage.
IMAGE_DIR = "/tmp/generated_images"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"

os.makedirs(IMAGE_DIR, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate_image():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Request must be JSON with a 'prompt' key."}), 400

    prompt = request.json['prompt']
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({"error": "Server configuration error: API key not found."}), 500

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    try:
        api_url_with_key = f"{GEMINI_API_URL}?key={api_key}"
        response = requests.post(api_url_with_key, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()
        
        image_part = next((p for p in result['candidates'][0]['content']['parts'] if 'inlineData' in p), None)
        if not image_part:
            return jsonify({"error": "No image data found in API response."}), 500
            
        image_data = base64.b64decode(image_part['inlineData']['data'])
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
        # IMPORTANT: We can't serve static files this way on Vercel.
        # We will return the image data directly in the response.
        # This is a more robust serverless pattern.
        
        # Instead of saving and returning a URL, we return the image directly
        encoded_string = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;base64,{encoded_string}"
        
        return jsonify({"imageUrl": image_data_url})

    except requests.exceptions.RequestException as e:
        error_detail = "Failed to communicate with the image generation service."
        if e.response is not None: error_detail = e.response.text
        return jsonify({"error": error_detail}), 502
    except (KeyError, IndexError):
        return jsonify({"error": "Invalid response from image generation service."}), 500

@app.route('/')
def index():
    return "Image Generation API Gateway is running on Vercel!"

