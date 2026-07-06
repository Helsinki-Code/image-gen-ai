import os
import json
import re
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# -------------------------
# Gemini client
# -------------------------
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# -------------------------
# Models
# -------------------------
ORCHESTRATOR_MODEL = "models/gemini-3-flash-preview"
IMAGE_MODEL = "models/gemini-3.1-flash-lite-image"

# -------------------------
# Tools for article analysis
# -------------------------
TOOLS = [
    {"type": "google_search"},
    {"type": "url_context"},
]

# -------------------------
# Configs
# -------------------------
ORCHESTRATION_CONFIG = {
    "temperature": 1.2,
    "max_output_tokens": 65536,
    "top_p": 0.95,
    "thinking_level": "high",
}

IMAGE_CONFIG = {
    "temperature": 1,
    "max_output_tokens": 65536,
    "top_p": 0.95,
    "thinking_level": "low",
    "image_config": {
        "image_size": "1K",
    },
}


# -------------------------
# Helpers
# -------------------------
def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and (
        url.startswith("http://") or url.startswith("https://")
    )


def extract_text_parts(interaction):
    texts = []
    for step in getattr(interaction, "steps", []):
        if getattr(step, "type", None) == "model_output" and getattr(step, "content", None):
            for part in step.content:
                if getattr(part, "type", None) == "text":
                    texts.append(part.text)
    return "\n".join(texts).strip()


def extract_image_parts(interaction):
    images = []
    texts = []

    for step in getattr(interaction, "steps", []):
        if getattr(step, "type", None) == "model_output" and getattr(step, "content", None):
            for part in step.content:
                if getattr(part, "type", None) == "image":
                    images.append(f"data:image/png;base64,{part.data}")
                elif getattr(part, "type", None) == "text":
                    texts.append(part.text)

    return images, texts


def parse_json_from_text(text: str):
    """
    Gemini may sometimes wrap JSON in markdown fences.
    This helper extracts the JSON object safely.
    """
    if not text:
        raise ValueError("Empty model response")

    # Remove markdown fences if present
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try parsing directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first big JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Could not find JSON object in model response.")

    return json.loads(match.group(0))


def build_orchestration_prompt(article_url: str, max_images: int):
    return f"""
You are an expert editorial visual strategist and article-image planning agent.

Your job:
1. Visit and analyze the full article at this URL: {article_url}
2. Understand the full article deeply.
3. Decide how many inline images the article needs, up to a maximum of {max_images}.
4. Decide the best placement point for each image.
5. Create extremely high-quality, highly detailed, editorial-grade, photorealistic image prompts for each placement.

The generated images must:
- feel premium and publication-quality
- be extremely realistic
- be contextually accurate to the article
- be visually compelling for article readers
- avoid looking like generic stock images when possible
- avoid text overlays inside the image
- avoid watermarks
- avoid surreal or obviously fake visual artifacts unless article context demands it

Return STRICT JSON only.
Do not include explanation outside JSON.

Use this exact schema:

{{
  "article_title": "string",
  "article_summary": "string",
  "overall_visual_strategy": "string",
  "recommended_image_count": 3,
  "images": [
    {{
      "image_number": 1,
      "section_heading": "string",
      "placement": "e.g. after introduction, after section 2, before conclusion",
      "placement_reason": "string",
      "purpose": "string",
      "aspect_ratio": "16:9",
      "alt_text": "string",
      "prompt": "A very detailed photorealistic editorial image prompt",
      "negative_prompt": "Avoid text, watermark, blur, distorted anatomy, extra fingers, bad composition, low detail, cartoon look"
    }}
  ]
}}

Rules:
- Return between 1 and {max_images} images.
- Prompt quality must be excellent and highly descriptive.
- Prompts must be optimized for realistic article imagery.
- Prompts must mention composition, lighting, realism, environment, subject details, mood, and publication-style quality.
- If article is abstract or conceptual, still create realistic editorial-style imagery.
"""
    

def build_image_generation_prompt(article_title: str, article_summary: str, item: dict):
    base_prompt = item.get("prompt", "")
    negative_prompt = item.get("negative_prompt", "")

    return f"""
Create a single extremely realistic, high-quality editorial article image.

Article title:
{article_title}

Article summary:
{article_summary}

Section heading:
{item.get("section_heading", "")}

Image purpose:
{item.get("purpose", "")}

Placement:
{item.get("placement", "")}

Main prompt:
{base_prompt}

Requirements:
- ultra realistic
- premium editorial article image
- highly detailed
- natural lighting or cinematic realistic lighting as appropriate
- accurate materials and textures
- authentic human anatomy if people are present
- visually rich but not exaggerated
- suitable for a high-quality online article
- no text inside the image
- no watermark
- no logo
- no collage
- single cohesive scene
- strong composition
- believable scene realism
- avoid obvious AI look

Negative guidance:
{negative_prompt}
"""


# -------------------------
# Main route
# -------------------------
@app.route("/generate-article-images", methods=["POST"])
def generate_article_images():
    """
    Request JSON:
    {
      "article_url": "https://example.com/article",
      "max_images": 3
    }
    """

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON."}), 400

        data = request.get_json()
        article_url = data.get("article_url")
        max_images = int(data.get("max_images", 3))

        if not article_url:
            return jsonify({"error": "Missing 'article_url'."}), 400

        if not is_valid_url(article_url):
            return jsonify({"error": "Invalid 'article_url'. Must start with http:// or https://"}), 400

        max_images = max(1, min(max_images, 8))

        # ---------------------------------------
        # Step 1: Analyze article and plan images
        # ---------------------------------------
        orchestration_prompt = build_orchestration_prompt(article_url, max_images)

        orchestration_interaction = client.interactions.create(
            model=ORCHESTRATOR_MODEL,
            input=orchestration_prompt,
            tools=TOOLS,
            generation_config=ORCHESTRATION_CONFIG,
        )

        orchestration_text = extract_text_parts(orchestration_interaction)
        plan = parse_json_from_text(orchestration_text)

        article_title = plan.get("article_title", "")
        article_summary = plan.get("article_summary", "")
        images_plan = plan.get("images", [])

        if not images_plan:
            return jsonify({
                "error": "The agent analyzed the article but did not return any image plan.",
                "raw_response": orchestration_text
            }), 500

        # ---------------------------------------
        # Step 2: Generate each image
        # ---------------------------------------
        generated_images = []

        for item in images_plan:
            prompt = build_image_generation_prompt(article_title, article_summary, item)

            image_interaction = client.interactions.create(
                model=IMAGE_MODEL,
                input=prompt,
                generation_config=IMAGE_CONFIG,
                response_modalities=["image", "text"],
            )

            image_urls, text_outputs = extract_image_parts(image_interaction)

            if not image_urls:
                generated_images.append({
                    "image_number": item.get("image_number"),
                    "section_heading": item.get("section_heading"),
                    "placement": item.get("placement"),
                    "alt_text": item.get("alt_text"),
                    "prompt": item.get("prompt"),
                    "status": "failed",
                    "error": "No image returned by model.",
                    "model_text": text_outputs,
                })
                continue

            generated_images.append({
                "image_number": item.get("image_number"),
                "section_heading": item.get("section_heading"),
                "placement": item.get("placement"),
                "placement_reason": item.get("placement_reason"),
                "purpose": item.get("purpose"),
                "aspect_ratio": item.get("aspect_ratio", "16:9"),
                "alt_text": item.get("alt_text"),
                "prompt": item.get("prompt"),
                "negative_prompt": item.get("negative_prompt"),
                "imageUrl": image_urls[0],
                "images": image_urls,
                "model_text": text_outputs,
                "status": "success",
            })

        return jsonify({
            "success": True,
            "article_url": article_url,
            "article_title": article_title,
            "article_summary": article_summary,
            "overall_visual_strategy": plan.get("overall_visual_strategy", ""),
            "recommended_image_count": plan.get("recommended_image_count", len(generated_images)),
            "image_plan": images_plan,
            "generated_images": generated_images,
        })

    except Exception as e:
        print(f"Error in /generate-article-images: {e}")
        return jsonify({
            "error": "Failed to analyze article and generate images.",
            "details": str(e)
        }), 500


# -------------------------
# Health route
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return "Article Image Agent is running!"
