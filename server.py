from pathlib import Path
import threading
import os

# Silence sklearn/pandas warnings during transformers import
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, jsonify, request, send_from_directory

import torch
import importlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
ADAPTER_PATH = "."

app = Flask(__name__, static_folder=None)

use_cuda = torch.cuda.is_available()

_model_lock = threading.Lock()
_model = None
_tokenizer = None
_model_error = None


def _build_bnb_config():
    if not use_cuda:
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_model():
    global _model, _tokenizer, _model_error
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer
        if _model_error is not None:
            raise RuntimeError(_model_error)

        try:
            bnb_config = _build_bnb_config()

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto" if use_cuda else "cpu",
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                low_cpu_mem_usage=use_cuda,
            )

            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            model.eval()
            model.generation_config.pad_token_id = tokenizer.eos_token_id

            _model = model
            _tokenizer = tokenizer
            return _model, _tokenizer
        except Exception as exc:
            _model_error = str(exc)
            raise

WEB_DIR = Path(__file__).parent / "web"


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(WEB_DIR, "styles.css")


@app.get("/app.js")
def app_js():
    return send_from_directory(WEB_DIR, "app.js")


@app.get("/api/status")
def status():
    if _model_error is not None:
        return jsonify({"status": "error", "detail": _model_error})
    if _model is None:
        return jsonify({"status": "loading"})
    return jsonify({"status": "ready"})


@app.post("/api/solve")
def solve():
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    if not question:
        return "Question is required", 400

    max_new_tokens = int(payload.get("max_new_tokens") or 200)

    prompt = (
        "### Instruction:\n"
        "Solve the math problem step by step and give the final answer.\n\n"
        "### Problem:\n"
        f"{question}\n\n"
        "### Solution:\n"
    )

    try:
        model, tokenizer = get_model()
    except Exception as exc:
        return f"Model failed to load: {exc}", 500

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"output": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
