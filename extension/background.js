// Service worker: relays the content script's rewrite request to the local Defender server.
// The model runs on 127.0.0.1 (your PC), so nothing leaves the machine.
const ENDPOINT = "http://127.0.0.1:8765/rewrite";

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "defender-warmup-req") {
    // optional: touch the server so it's awake; ignore failures
    fetch("http://127.0.0.1:8765/health").catch(() => {});
    sendResponse({ ok: true });
    return true;
  }
  if (msg.type === "defender-rewrite-req") {
    fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: msg.text }),
    })
      .then((r) => r.json())
      .then((d) => sendResponse({ ok: true, text: d.rewrite }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true; // async
  }
});
