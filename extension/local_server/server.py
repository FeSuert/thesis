"""Local on-device Defender server.

Loads the merged Qwen3.5 Defender with native transformers (the only stack that supports
this architecture) and exposes POST /rewrite {"text": ...} -> {"rewrite": ...} on 127.0.0.1.
The browser extension calls this; nothing leaves the machine.

Run:
    pip install flask transformers accelerate torch          # (+ bitsandbytes for --mode 4bit)
    python server.py --model /path/to/merged --port 8765 --mode auto

--mode:
    auto  (default) bf16 on GPU with CPU offload for whatever doesn't fit (safe on 8GB).
    4bit  bitsandbytes nf4 on GPU (fastest if bitsandbytes works on your GPU).
    cpu   fp32 on CPU only (slowest, but rock-solid; you have the RAM for it).
"""

import argparse
import torch
from flask import Flask, request, jsonify

# Exact Defender system prompt (from models/defender/defender.py).
DEFENDER_SYSTEM = (
    "You are a privacy-preserving rewriting assistant. Rewrite the user's message so it can be "
    "safely sent to an external AI chatbot without revealing the user's personal identity. Replace "
    "or remove specific personal details — location, profession, age, sex/gender, relationship "
    "or family status, and income/socioeconomic status — with more general expressions, while "
    "fully preserving the original meaning, intent, and natural fluency. Generalize details that are "
    "needed to answer the request (e.g. \"nurse in Boston\" → \"healthcare worker\"); drop "
    "details that are irrelevant to the request. Never insert false information, never leave blanks, "
    "placeholders, or bracketed tags, and never add explanations. If the message contains no "
    "identifying details, return it essentially unchanged. Output only the rewritten message."
)


def load(model_dir, mode):
    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if mode == "4bit":
        from transformers import BitsAndBytesConfig
        kw.update(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    elif mode == "cpu":
        kw.update(torch_dtype=torch.float32, device_map={"": "cpu"})
    else:  # auto: bf16, GPU + CPU offload
        kw.update(torch_dtype=torch.bfloat16, device_map="auto")

    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_dir, **kw)
            print(f"[server] loaded with {cls_name}, mode={mode}")
            model.eval()
            return tok, model
        except Exception as e:  # noqa
            print(f"[server] {cls_name} failed: {e}")
            last = e
    raise last


def build_app(tok, model):
    app = Flask(__name__)

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(ok=True)

    @app.route("/rewrite", methods=["POST", "OPTIONS"])
    def rewrite():
        if request.method == "OPTIONS":
            return ("", 204)
        text = ((request.get_json(force=True, silent=True) or {}).get("text") or "").strip()
        if not text:
            return jsonify(rewrite="")
        messages = [
            {"role": "system", "content": DEFENDER_SYSTEM},
            {"role": "user", "content": text},
        ]
        # This model's apply_chat_template returns a BatchEncoding (dict with input_ids +
        # attention_mask), not a bare tensor — so ask for a dict and unpack it into generate().
        common = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(messages, enable_thinking=False, **common)
        except TypeError:
            enc = tok.apply_chat_template(messages, **common)
        try:
            enc = enc.to(model.device)
        except AttributeError:
            enc = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return jsonify(rewrite=gen)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to the merged Defender model dir.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mode", choices=["auto", "4bit", "cpu"], default="auto")
    args = p.parse_args()

    print(f"[server] loading {args.model} (mode={args.mode}) ...")
    tok, model = load(args.model, args.mode)
    print(f"[server] ready on http://127.0.0.1:{args.port}")
    build_app(tok, model).run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
