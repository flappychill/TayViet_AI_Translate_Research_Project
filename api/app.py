from flask import Flask, request, render_template, jsonify
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_name():
    p = os.path.join(os.path.dirname(__file__), "..", "config", "model.toml")
    data = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" in s:
                k,v = s.split("=",1)
                data[k.strip()] = v.strip().strip('"').strip("'")
    return data

cfg = load_name()
app = Flask(__name__)

def pick_device():
    d = os.environ.get("DEVICE", cfg.get("device","auto"))
    if d == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = pick_device()
name_a = os.environ.get("MODEL_TAY2VI", cfg.get("model_tay2vi"))
name_b = os.environ.get("MODEL_VI2TAY", cfg.get("model_vi2tay"))

tok_a = AutoTokenizer.from_pretrained(name_a)
tok_b = AutoTokenizer.from_pretrained(name_b)
mod_a = AutoModelForSeq2SeqLM.from_pretrained(name_a).to(device)
mod_b = AutoModelForSeq2SeqLM.from_pretrained(name_b).to(device)
max_new = int(os.environ.get("MAX_NEW_TOKENS", cfg.get("max_new_tokens","96")))
beam = int(os.environ.get("BEAM", cfg.get("beam","4")))

def translate_one(text, direction):
    if direction == "tay2vi":
        ids = tok_a(text, return_tensors="pt", truncation=True).to(device)
        out = mod_a.generate(**ids, max_new_tokens=max_new, num_beams=beam)
        return tok_a.decode(out[0], skip_special_tokens=True)
    if direction == "vi2tay":
        ids = tok_b(text, return_tensors="pt", truncation=True).to(device)
        out = mod_b.generate(**ids, max_new_tokens=max_new, num_beams=beam)
        return tok_b.decode(out[0], skip_special_tokens=True)
    return ""

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    direction = (data.get("direction") or "tay2vi").strip()
    if not text:
        return jsonify({"ok": True, "text": "", "translation": "", "direction": direction})
    return jsonify({"ok": True, "text": text, "translation": translate_one(text, direction), "direction": direction})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)

@app.route("/api/translate_batch", methods=["POST"])
def api_translate_batch():
    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    direction = (data.get("direction") or "tay2vi").strip()
    outs = []
    for it in items:
        t = str(it or "").strip()
        if not t:
            outs.append("")
        else:
            outs.append(translate_one(t, direction))
    return jsonify({"ok": True, "direction": direction, "translations": outs})
