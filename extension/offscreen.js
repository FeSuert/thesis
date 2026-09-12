// Runs the on-device rewriter with WebLLM (WebGPU). Loaded only in the offscreen document.
// This is the SOURCE file. Bundle it into offscreen.bundle.js with esbuild (see setup notes);
// offscreen.html loads the bundle, not this file directly.
import * as webllm from "@mlc-ai/web-llm";

// STOCK model for the plumbing test. GTX 10xx (Pascal) has no shader-f16, so use a *q4f32_1* build.
// Later we swap this for the compiled Defender (e.g. "Qwen3.5-4B-Defender-q4f32_1-MLC").
const MODEL_ID = "Llama-3.2-1B-Instruct-q4f32_1-MLC";

// Exact Defender system prompt (from models/defender/defender.py) so behaviour matches the thesis model.
const DEFENDER_SYSTEM =
  "You are a privacy-preserving rewriting assistant. Rewrite the user's message so it can be " +
  "safely sent to an external AI chatbot without revealing the user's personal identity. Replace " +
  "or remove specific personal details — location, profession, age, sex/gender, relationship " +
  "or family status, and income/socioeconomic status — with more general expressions, while " +
  "fully preserving the original meaning, intent, and natural fluency. Generalize details that are " +
  "needed to answer the request (e.g. \"nurse in Boston\" → \"healthcare worker\"); drop " +
  "details that are irrelevant to the request. Never insert false information, never leave blanks, " +
  "placeholders, or bracketed tags, and never add explanations. If the message contains no " +
  "identifying details, return it essentially unchanged. Output only the rewritten message.";

let enginePromise = null;
function getEngine() {
  if (!enginePromise) {
    console.log("[Defender/offscreen] loading model:", MODEL_ID);
    enginePromise = webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (r) => console.log("[Defender/offscreen] load:", r.text),
    });
  }
  return enginePromise;
}

async function rewrite(text) {
  const engine = await getEngine();
  const out = await engine.chat.completions.create({
    messages: [
      { role: "system", content: DEFENDER_SYSTEM },
      { role: "user", content: text },
    ],
    temperature: 0, // greedy, matches the evaluation
    max_tokens: 512,
  });
  return out.choices[0].message.content.trim();
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "defender-warmup") { getEngine(); sendResponse({ ok: true }); return true; }
  if (msg.type === "defender-rewrite") {
    rewrite(msg.text).then(
      (text) => sendResponse({ ok: true, text }),
      (err) => { console.warn("[Defender/offscreen] error:", err); sendResponse({ ok: false, error: String(err) }); }
    );
    return true; // async: keep the message channel open
  }
});
