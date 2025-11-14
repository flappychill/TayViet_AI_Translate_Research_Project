from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app = Flask(__name__)
model_name = "IAmSkyDra/BARTTay_Translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
@app.route("/", methods=["GET","POST"])
def index():
    output = ""
    if request.method == "POST":
        text = request.form.get("text","").strip()
        if text:
            ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
            out = model.generate(ids)
            output = tokenizer.decode(out[0], skip_special_tokens=True)
    return render_template("index.html", output=output)
@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = data.get("text","").strip()
    if not text:
        return jsonify({"text": "", "translation": ""})
    ids = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
    out = model.generate(ids)
    translation = tokenizer.decode(out[0], skip_special_tokens=True)
    return jsonify({"text": text, "translation": translation})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
