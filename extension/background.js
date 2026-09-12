// Service worker: owns the offscreen document and relays messages content-script <-> offscreen.
let creating = null;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["WORKERS"],
      justification: "Run the on-device privacy rewriter model with WebGPU.",
    });
  }
  await creating;
  creating = null;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "defender-warmup-req") {
    (async () => {
      await ensureOffscreen();
      await chrome.runtime.sendMessage({ type: "defender-warmup" });
      sendResponse({ ok: true });
    })().catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }
  if (msg.type === "defender-rewrite-req") {
    (async () => {
      await ensureOffscreen();
      const res = await chrome.runtime.sendMessage({ type: "defender-rewrite", text: msg.text });
      sendResponse(res);
    })().catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true; // async
  }
});
