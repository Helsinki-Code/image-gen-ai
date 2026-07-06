import os
import base64
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

MODEL_NAME = "models/gemini-3.1-flash-lite-image"


@app.route("/generate", methods=["POST"])
def generate_image():
    """
    Generate an image using Gemini 3.1 Flash Lite Image model.
    Request body:
    {
      "prompt": "A beautiful sunset over ocean"
    }
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "Server configuration error: GEMINI_API_KEY not found."
        }), 500

    if not request.is_json:
        return jsonify({
            "error": "Request must be JSON."
        }), 400

    data = request.get_json()

    prompt = data.get("prompt")

    if not prompt:
        return jsonify({
            "error": "Request JSON must include a non-empty 'prompt' field."
        }), 400

    try:
        client = genai.Client(api_key=api_key)

        generation_config = {
            "temperature": data.get("temperature", 1),
            "max_output_tokens": data.get("max_output_tokens", 65536),
            "top_p": data.get("top_p", 0.95),
            "thinking_level": data.get("thinking_level", "minimal"),
            "image_config": {
                "image_size": data.get("image_size", "1K"),
            },
        }

        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            generation_config=generation_config,
            response_modalities=["image", "text"],
        )

        image_urls = []
        text_outputs = []

        for step in interaction.steps:
            if step.type == "model_output" and step.content:
                for part in step.content:
                    if part.type == "text":
                        text_outputs.append(part.text)

                    elif part.type == "image":
                        image_url = f"data:image/png;base64,{part.data}"
                        image_urls.append(image_url)

        if not image_urls:
            return jsonify({
                "error": "No image data found in Gemini response.",
                "text": text_outputs
            }), 500

        return jsonify({
            "success": True,
            "model": MODEL_NAME,
            "imageUrl": image_urls[0],
            "images": image_urls,
            "text": text_outputs
        })

    except Exception as e:
        print(f"Gemini image generation error: {e}")

        return jsonify({
            "error": "Image generation failed.",
            "details": str(e)
        }), 502


@app.route("/", methods=["GET"])
def index():
    return "Gemini 3.1 Flash Lite Image API is running!"
