import os
import base64
import requests
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# The official REST API endpoint for the Nano Banana model
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"

@app.route('/generate', methods=['POST'])
def generate_image_gemini():
    """
    Handles a request to generate an image using the Gemini text-to-image model.
    """
    # 1. Get the API Key from Vercel's environment variables
    # Note: We are using GEMINI_API_KEY for consistency.
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return jsonify({"error": "Server configuration error: API key not found."}), 500

    # 2. Validate the incoming request
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Request must be JSON with a 'prompt' key."}), 400
    prompt = request.json['prompt']
    if not prompt:
        return jsonify({"error": "The 'prompt' cannot be empty."}), 400

    # 3. Construct the payload for the Gemini API
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

    # 4. Make the request to the Gemini API
    try:
        api_url_with_key = f"{GEMINI_API_URL}?key={api_key}"
        response = requests.post(api_url_with_key, json=payload, timeout=45)
        response.raise_for_status()  # This will raise an exception for HTTP error codes
        result = response.json()

        # 5. Extract the Base64 image data from the response
        # We search through the response parts to find the one with image data.
        image_part = next((p for p in result['candidates'][0]['content']['parts'] if 'inlineData' in p), None)
        if not image_part:
            return jsonify({"error": "No image data found in API response."}), 500
        
        base64_data = image_part['inlineData']['data']
        
        # 6. Format as a Data URL and return the response
        # This is a robust way to send an image without needing to save files.
        image_url = f"data:image/png;base64,{base64_data}"
        return jsonify({"imageUrl": image_url})

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        # Try to return the actual error from the API if possible
        error_detail = "Failed to communicate with the image generation service."
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = e.response.text
        return jsonify({"error": error_detail}), 502
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}")
        return jsonify({"error": "Invalid response format from image generation service."}), 500

@app.route('/')
def index():
    return "Gemini Image Generation API is running on Vercel!"

