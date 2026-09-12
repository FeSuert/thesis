// Content-side bridge: same __defenderRewrite(text) contract as the old mock, but routes the
// message to the extension's offscreen WebLLM engine (background -> offscreen -> back).
window.__defenderRewrite = function (text) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type: "defender-rewrite-req", text }, (res) => {
      if (chrome.runtime.lastError) return reject(chrome.runtime.lastError.message);
      if (!res || !res.ok) return reject((res && res.error) || "no response");
      resolve(res.text);
    });
  });
};

// Start downloading/loading the model as soon as a supported page opens, so the first
// real message isn't stuck waiting for the whole model to load.
try { chrome.runtime.sendMessage({ type: "defender-warmup-req" }); } catch (e) {}
