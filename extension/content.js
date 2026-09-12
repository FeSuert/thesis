(() => {
  // --- dev flags ---
  const DEV_ALWAYS_REVIEW = true; // show the popup even when the rewrite == original (so you can see interception working)

  // Candidate selectors for the message box, tried in order. Update if a site changes its DOM.
  const COMPOSER_SELECTORS = [
    "#prompt-textarea",                          // ChatGPT (contenteditable)
    'div.ProseMirror[contenteditable="true"]',   // Claude
    'div[contenteditable="true"]',
    "textarea",
  ];
  // Candidate selectors for the send button.
  const SEND_SELECTORS = [
    '[data-testid="send-button"]',               // ChatGPT
    'button[aria-label="Send message"]',         // Claude
    'button[aria-label*="Send"]',
  ];

  let overlayOpen = false;
  const isEditable = (el) => el && (el.tagName === "TEXTAREA" || el.isContentEditable);
  const getText = (el) => (el.tagName === "TEXTAREA" ? el.value : el.innerText);

  function findComposer(target) {
    if (isEditable(target)) return target;
    for (const s of COMPOSER_SELECTORS) { const el = document.querySelector(s); if (el) return el; }
    return null;
  }

  // Write text back so the site's framework (React/ProseMirror) actually registers it.
  function setText(el, text) {
    el.focus();
    if (el.tagName === "TEXTAREA") {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
      setter.call(el, text);                      // bypass React's value shadowing
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      const sel = window.getSelection(), range = document.createRange();
      range.selectNodeContents(el); sel.removeAllRanges(); sel.addRange(range);
      document.execCommand("insertText", false, text); // reliable for contenteditable/ProseMirror
    }
  }

  function doSend() {
    for (const s of SEND_SELECTORS) {
      const b = document.querySelector(s);
      if (b && !b.disabled) { b.click(); return true; }
    }
    console.warn("[Defender] no send button found; text is in the box, press Enter yourself.");
    return false;
  }

  // Modal: shows the (editable) rewrite, hides the original behind a toggle, returns the user's choice.
  function showReview(original, rewrite) {
    return new Promise((resolve) => {
      overlayOpen = true;
      const wrap = document.createElement("div");
      wrap.style.cssText =
        "position:fixed;inset:0;z-index:2147483647;background:rgba(0,0,0,.45);display:flex;" +
        "align-items:center;justify-content:center;font:14px system-ui,sans-serif";
      wrap.innerHTML = `
        <div style="background:#fff;color:#111;max-width:640px;width:92%;border-radius:12px;padding:20px;box-shadow:0 10px 40px rgba(0,0,0,.3)">
          <div style="font-weight:600;margin-bottom:10px">Review before sending</div>
          <div style="font-size:12px;color:#666;margin-bottom:4px">Rewritten (editable, this is what gets sent)</div>
          <div id="dfn-rw" contenteditable="true" style="border:1px solid #ccc;border-radius:8px;padding:10px;white-space:pre-wrap;max-height:180px;overflow:auto;margin-bottom:12px"></div>
          <details style="margin-bottom:14px"><summary style="cursor:pointer;color:#666;font-size:12px">Show original</summary>
            <div id="dfn-orig" style="border:1px solid #eee;border-radius:8px;padding:10px;white-space:pre-wrap;max-height:160px;overflow:auto;margin-top:8px;color:#444"></div>
          </details>
          <div style="display:flex;gap:8px;justify-content:flex-end">
            <button id="dfn-orig-btn" style="padding:8px 12px;border:1px solid #ccc;background:#f5f5f5;border-radius:8px;cursor:pointer">Send original</button>
            <button id="dfn-cancel" style="padding:8px 12px;border:1px solid #ccc;background:#f5f5f5;border-radius:8px;cursor:pointer">Cancel</button>
            <button id="dfn-send" style="padding:8px 12px;border:0;background:#10a37f;color:#fff;border-radius:8px;cursor:pointer">Send rewritten</button>
          </div>
        </div>`;
      document.body.appendChild(wrap);
      wrap.querySelector("#dfn-rw").textContent = rewrite;
      wrap.querySelector("#dfn-orig").textContent = original;
      const close = (r) => { wrap.remove(); overlayOpen = false; resolve(r); };
      wrap.querySelector("#dfn-send").onclick = () => close({ action: "send", text: wrap.querySelector("#dfn-rw").textContent });
      wrap.querySelector("#dfn-orig-btn").onclick = () => close({ action: "send", text: original });
      wrap.querySelector("#dfn-cancel").onclick = () => close({ action: "cancel" });
    });
  }

  // Intercept Enter in the capture phase, before the site's own handler runs.
  document.addEventListener("keydown", async (e) => {
    if (overlayOpen || e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    const composer = findComposer(e.target);
    if (!composer || !getText(composer).trim()) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    const original = getText(composer).trim();
    console.log("[Defender] intercepted Enter. Original:", original);
    let rewrite = original;
    try { rewrite = await window.__defenderRewrite(original); }
    catch (err) { console.warn("[Defender] rewrite failed, using original:", err); }
    console.log("[Defender] rewrite:", rewrite);
    if (rewrite.trim() === original && !DEV_ALWAYS_REVIEW) { doSend(); return; }
    const res = await showReview(original, rewrite);
    if (res.action === "send") { setText(composer, res.text); setTimeout(doSend, 40); }
    // cancel -> leave the composer as-is for the user to edit
  }, true); // <-- capture phase

  console.log("[Defender] content script active on", location.hostname);
})();
