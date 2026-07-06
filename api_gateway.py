import os
import json
import re
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# ── Models ────────────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL = "models/gemini-3-flash-preview"       # analysis + planning brain
IMAGE_MODEL = "models/gemini-3.1-flash-lite-image"         # image generation engine


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, jsonify({"error": "Server configuration error: GEMINI_API_KEY not found."}), 500
    return genai.Client(api_key=api_key), None, None


def extract_text_from_interaction(interaction):
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
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise ValueError("No valid JSON object found in orchestrator output.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — ORCHESTRATOR: read + analyze the article, plan image placements.
# The agent itself decides HOW MANY images the article needs.
# ══════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert editorial art director for premium online publications.

You will be given the URL of an article. Use your url_context tool to read the FULL article
(and google_search only if you need extra context about entities mentioned in it).

Your job:
1. Deeply analyze the article: topic, narrative arc, tone, target audience, structure, and length.
2. YOU decide how many images this specific article genuinely needs — based on its length,
   number of sections, and visual storytelling value. A short article may need only 1-2 images;
   a long multi-section guide may justify many more. Do NOT pad; every image must earn its place.
3. Decide EXACTLY where each image should be placed (hero, section breaks, concept
   illustrations, step/exercise visuals, closing image).
4. For EACH placement, write a highly detailed, production-grade image generation prompt.

PROMPT QUALITY RULES (critical):
- Every prompt must produce a photorealistic / extremely realistic, high-quality editorial image.
- Include: subject, setting, composition, camera angle, lens (e.g. 35mm, 85mm f/1.4),
  lighting (golden hour, softbox, overcast), color palette, mood, texture and depth-of-field details.
- Match the article's tone (health/fitness = bright authentic candid photography; serious news =
  documentary style; tech = clean modern).
- NEVER include real people's names, celebrity likenesses, logos, brand marks, or any text/words to render in the image.
- Each prompt must be self-contained (the image model will NOT see the article).

Respond with ONLY a valid JSON object, no markdown fences, in this exact schema:

{
  "article_title": "string",
  "article_summary": "2-3 sentence summary",
  "tone": "string",
  "recommended_image_count": <integer you decided>,
  "reasoning": "1-2 sentences on why this number of images fits this article",
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
}"""


def run_orchestrator(client, article_url, max_images=None):
    tools = [
        {"type": "google_search"},
        {"type": "url_context"},
    ]
    generation_config = {
        "temperature": 1.0,
        "max_output_tokens": 65536,
        "top_p": 0.95,
        "thinking_level": "high",
    }
    user_input = ORCHESTRATOR_SYSTEM_PROMPT + f"\n\nArticle URL to analyze: {article_url}"
    if max_images:  # only constrain if the caller explicitly asked for a ceiling
        user_input += f"\nThe caller has set a hard ceiling of {max_images} images. Stay at or under it."

    interaction = client.interactions.create(
        model=ORCHESTRATOR_MODEL,
        input=user_input,
        tools=tools,
        generation_config=generation_config,
    )
    plan = parse_json_block(extract_text_from_interaction(interaction))

    if "images" not in plan or not isinstance(plan["images"], list) or not plan["images"]:
        raise ValueError("Orchestrator returned no image plan.")
    if max_images:
        plan["images"] = plan["images"][:max_images]
    return plan


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — IMAGE ENGINE: generate each planned image
# ══════════════════════════════════════════════════════════════════════════

def generate_single_image(client, prompt, image_size="2K"):
    """Returns raw base64 string (no data-URL prefix) or None."""
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
                if part.type == "image" and part.data:
                    data = part.data
                    # Normalize: SDK may return bytes or str; strip any data-URL prefix
                    if isinstance(data, bytes):
                        import base64 as _b64
                        data = _b64.b64encode(data).decode("ascii")
                    if data.startswith("data:"):
                        data = data.split(",", 1)[1]
                    return data
    return None


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/generate-article-images", methods=["POST"])
def generate_article_images():
    """
    Full agent pipeline: URL in → analyzed article → dynamic placement plan → generated images.

    Request body:
    {
      "url": "https://example.com/some-article",   // required
      "image_size": "2K",                          // optional: "1K" | "2K" | "4K"
      "max_images": 6,                             // optional ceiling; omit to let the agent decide freely
      "plan_only": false                           // optional: true = plan without generating
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

    max_images = data.get("max_images")  # None = agent decides dynamically
    if max_images is not None:
        max_images = max(1, int(max_images))
    image_size = data.get("image_size", "2K")
    plan_only = bool(data.get("plan_only", False))

    # ---- Step 1: analyze + plan (agent decides the count) ----
    try:
        plan = run_orchestrator(client, article_url, max_images)
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return jsonify({"error": "Article analysis / planning failed.", "details": str(e)}), 502

    if plan_only:
        return jsonify({
            "success": True,
            "url": article_url,
            "plan": plan,
            "images_generated": 0,
        })

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
            "mime_type": "image/png",
            "image_base64": None,   # raw base64 — decode this directly to get the PNG
            "imageUrl": None,       # same data as a data-URL, for direct <img src=...> use
            "status": "failed",
        }
        try:
            b64 = generate_single_image(client, item["image_prompt"], image_size=image_size)
            if b64:
                entry["image_base64"] = b64
                entry["imageUrl"] = f"data:image/png;base64,{b64}"
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
        "recommended_image_count": plan.get("recommended_image_count", len(plan["images"])),
        "agent_reasoning": plan.get("reasoning"),
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "image_model": IMAGE_MODEL,
        "images_planned": len(plan["images"]),
        "images_generated": generated,
        "images": results,
    }), status_code


@app.route("/generate", methods=["POST"])
def generate_image():
    """Legacy single-prompt image endpoint."""
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
        b64 = generate_single_image(client, prompt, image_size=data.get("image_size", "1K"))
        if not b64:
            return jsonify({"error": "No image data found in Gemini response."}), 500
        return jsonify({
            "success": True,
            "model": IMAGE_MODEL,
            "mime_type": "image/png",
            "image_base64": b64,
            "imageUrl": f"data:image/png;base64,{b64}",
        })
    except Exception as e:
        print(f"Gemini image generation error: {e}")
        return jsonify({"error": "Image generation failed.", "details": str(e)}), 502


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Article Image Agent",
        "description": "POST an article URL → agent reads it, decides how many images it needs, "
                       "plans placements, writes detailed prompts, and generates hyper-realistic editorial images.",
        "endpoints": {
            "POST /generate-article-images": {"url": "required", "image_size": "2K", "max_images": "optional ceiling", "plan_only": False},
            "POST /generate": {"prompt": "required (legacy direct image gen)"},
        },
        "orchestrator_model": ORCHESTRATOR_MODEL,
        "image_model": IMAGE_MODEL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
