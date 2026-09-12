# Browser Extension — On-Device Prompt Rewriter (Defender)

Rewrites your chat message with the thesis Defender model **before** it is sent
to ChatGPT / Claude. The model runs locally (127.0.0.1); nothing leaves the machine.

## Flow

```
content.js  →  rewriter.js  →  background.js  →  HTTP 127.0.0.1:8765  →  local_server/server.py (Defender)
```

- `content.js`      – intercepts the composer + Enter, shows the review popup (DEV_ALWAYS_REVIEW).
- `rewriter.js`     – page hook; fires `defender-rewrite-req` / `defender-warmup-req`.
- `background.js`   – service worker; POSTs the text to the local server, returns the rewrite.
- `manifest.json`   – MV3; host_permissions for 127.0.0.1:8765; content scripts on chatgpt.com / claude.ai.
- `local_server/server.py` – Flask server that loads the merged Qwen3.5 Defender via native
  transformers and serves POST /rewrite.

That is the complete file set. Nothing else is required.

## Run

1. Start the server (on this machine for a true on-device demo, or on Elysium + SSH tunnel to test):

   ```
   pip install flask transformers accelerate torch      # (+ bitsandbytes for --mode 4bit)
   python local_server/server.py --model /path/to/merged --port 8765 --mode auto
   ```

   `--mode`: `auto` (bf16 GPU + CPU offload), `4bit` (bitsandbytes nf4), `cpu` (fp32, slow but reliable).

2. Load the extension in `chrome://extensions` (Developer mode → Load unpacked), then reload the chat tab.

## Notes

- Earlier WebGPU/WebLLM in-browser path was dropped: the Defender's `qwen3_5` architecture
  (multimodal, linear-attention/SSM hybrid) is unsupported by browser inference runtimes.
