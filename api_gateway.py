import os
import json
import re
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# ── Models ────────────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL = "models/gemini-3-flash-preview"       # analysis + planning brain
IMAGE_MODEL = "models/gemini-3.1-flash-lite-image"         # image generation engine

MAX_IMAGES = 8  # hard cap so a long article doesn't burn your quota


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, jsonify({"error": "Server configuration error: GEMINI_API_KEY not found."}), 500
    return genai.Client(api_key=api_key), None, None


def extract_text_from_interaction(interaction):
    """Collect all text parts from every model_output step."""
    texts = []
    for step in interaction.steps:
        if step.type == "model_output" and step.content:
            for part in step.content:
                if part.type == "text" and part.text:
                    texts.append(part.text)
    return "\n".join(texts)


def parse_json_block(text):
    """Robustly pull a JSON object out of model output (handles ```json fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost { ... }
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise ValueError("No valid JSON object found in orchestrator output.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — ORCHESTRATOR: navigate + analyze the article, plan image placements
# ══════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert editorial art director for premium online publications.

You will be given the URL of an article. Use your url_context tool to read the FULL article
(and google_search only if you need extra context about entities mentioned in it).

Your job:
1. Deeply analyze the article: topic, narrative arc, tone, target audience, key sections.
2. Decide EXACTLY where images should be placed for maximum reader engagement
   (hero image, section breaks, concept illustrations, data/process visuals, closing image).
3. For EACH placement, write a highly detailed, production-grade image generation prompt.

PROMPT QUALITY RULES (critical):
- Every prompt must produce a photorealistic / extremely realistic, high-quality editorial image.
- Include: subject, setting, composition, camera angle, lens (e.g. 35mm, 85mm f/1.4),
  lighting (golden hour, softbox, overcast), color palette, mood, texture and depth-of-field details.
- Match the article's tone (serious news = documentary photography; lifestyle = warm candid; tech = clean modern).
- NEVER include real people's names, celebrity likenesses, logos, brand marks, or any text/words to render in the image.
- Each prompt must be self-contained (the image model will NOT see the article).

Respond with ONLY a valid JSON object, no markdown fences, in this exact schema:

{
  "article_title": "string",
  "article_summary": "2-3 sentence summary",
  "tone": "string",
  "images": [
    {
      "id": 1,
      "placement": "hero | after_section:<section heading> | after_paragraph:<first 8 words of the paragraph> | closing",
      "purpose": "why an image belongs here",
      "alt_text": "concise accessible alt text",
      "caption": "suggested caption for the article",
      "aspect_ratio": "16:9 | 4:3 | 1:1 | 3:4",
      "image_prompt": "the full detailed generation prompt"
    }
  ]
}

Plan between 3 and 6 images unless the article is very short (then 1-2)."""


def run_orchestrator(client, article_url, max_images):
    tools = [
        {"type": "google_search"},
        {"type": "url_context"},
    ]
    generation_config = {
        "temperature": 1.0,          # planning needs consistency more than wild creativity
        "max_output_tokens": 65536,
        "top_p": 0.95,
        "thinking_level": "high",
    }
    interaction = client.interactions.create(
        model=ORCHESTRATOR_MODEL,
        input=(
            ORCHESTRATOR_SYSTEM_PROMPT
            + f"\n\nArticle URL to analyze: {article_url}"
            + f"\nMaximum number of images allowed: {max_images}"
        ),
        tools=tools,
        generation_config=generation_config,
    )
    raw_text = extract_text_from_interaction(interaction)
    plan = parse_json_block(raw_text)

    if "images" not in plan or not isinstance(plan["images"], list) or not plan["images"]:
        raise ValueError("Orchestrator returned no image plan.")
    plan["images"] = plan["images"][:max_images]
    return plan


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — IMAGE ENGINE: generate each planned image
# ══════════════════════════════════════════════════════════════════════════

def generate_single_image(client, prompt, image_size="2K"):
    generation_config = {
        "temperature": 1,
        "max_output_tokens": 65536,
        "top_p": 0.95,
        "thinking_level": "low",
        "image_config": {
            "image_size": image_size,
        },
    }
    interaction = client.interactions.create(
        model=IMAGE_MODEL,
        input=prompt,
        generation_config=generation_config,
        response_modalities=["image", "text"],
    )
    for step in interaction.steps:
        if step.type == "model_output" and step.content:
            for part in step.content:
                if part.type == "image":
                    return f"data:image/png;base64,{part.data}"
    return None


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/generate-article-images", methods=["POST"])
def generate_article_images():
    """
    Full agent pipeline: URL in → analyzed article → placement plan → generated images out.

    Request body:
    {
      "url": "https://example.com/some-article",
      "max_images": 5,          // optional, default 5, capped at MAX_IMAGES
      "image_size": "2K",       // optional: "1K" | "2K" | "4K"
      "plan_only": false        // optional: true = return the plan without generating images
    }
    """
    client, err_resp, err_code = get_client()
    if err_resp:
        return err_resp, err_code

    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()
    article_url = (data.get("url") or "").strip()
    if not article_url.startswith(("http://", "https://")):
        return jsonify({"error": "Request JSON must include a valid 'url' field (http/https)."}), 400

    max_images = min(int(data.get("max_images", 5)), MAX_IMAGES)
    image_size = data.get("image_size", "2K")
    plan_only = bool(data.get("plan_only", False))

    # ---- Step 1: analyze + plan ----
    try:
        plan = run_orchestrator(client, article_url, max_images)
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return jsonify({"error": "Article analysis / planning failed.", "details": str(e)}), 502

    if plan_only:
        return jsonify({"success": True, "url": article_url, "plan": plan, "images_generated": 0})

    # ---- Step 2: generate each planned image ----
    results = []
    generated = 0
    for item in plan["images"]:
        entry = {
            "id": item.get("id"),
            "placement": item.get("placement"),
            "purpose": item.get("purpose"),
            "alt_text": item.get("alt_text"),
            "caption": item.get("caption"),
            "aspect_ratio": item.get("aspect_ratio"),
            "image_prompt": item.get("image_prompt"),
            "imageUrl": None,
            "status": "failed",
        }
        try:
            image_url = generate_single_image(client, item["image_prompt"], image_size=image_size)
            if image_url:
                entry["imageUrl"] = image_url
                entry["status"] = "ok"
                generated += 1
            else:
                entry["error"] = "No image data returned by model."
        except Exception as e:
            print(f"Image generation error (id={item.get('id')}): {e}")
            entry["error"] = str(e)
        results.append(entry)

    status_code = 200 if generated > 0 else 502
    return jsonify({
        "success": generated > 0,
        "url": article_url,
        "article_title": plan.get("article_title"),
        "article_summary": plan.get("article_summary"),
        "tone": plan.get("tone"),
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "image_model": IMAGE_MODEL,
        "images_planned": len(plan["images"]),
        "images_generated": generated,
        "images": results,
    }), status_code


@app.route("/generate", methods=["POST"])
def generate_image():
    """Legacy single-prompt image endpoint (kept for direct prompt → image use)."""
    client, err_resp, err_code = get_client()
    if err_resp:
        return err_resp, err_code

    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Request JSON must include a non-empty 'prompt' field."}), 400

    try:
        image_url = generate_single_image(client, prompt, image_size=data.get("image_size", "1K"))
        if not image_url:
            return jsonify({"error": "No image data found in Gemini response."}), 500
        return jsonify({
            "success": True,
            "model": IMAGE_MODEL,
            "imageUrl": image_url,
            "images": [image_url],
        })
    except Exception as e:
        print(f"Gemini image generation error: {e}")
        return jsonify({"error": "Image generation failed.", "details": str(e)}), 502


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Article Image Agent",
        "description": "POST an article URL → agent reads it, plans image placements, "
                       "writes detailed prompts, and generates hyper-realistic editorial images.",
        "endpoints": {
            "POST /generate-article-images": {"url": "required", "max_images": 5, "image_size": "2K", "plan_only": False},
            "POST /generate": {"prompt": "required (legacy direct image gen)"},
        },
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "image_model": IMAGE_MODEL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
