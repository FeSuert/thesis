// Mock Defender for shell development. Swapped for the WebLLM offscreen call later.
// Interface contract (kept stable): async function that maps a message string to a rewrite string.
window.__defenderRewrite = async function (text) {
  let r = text;
  const rules = [
    [/\bin ([A-Z][a-zA-Zé]+)\b/g, "in a city"],
    [/\bmy (husband|wife|boyfriend|girlfriend)\b/gi, "my partner"],
    [/\bas an? ([a-z]+ )?(nurse|doctor|engineer|teacher|lawyer|developer)\b/gi, "as a professional"],
  ];
  for (const [re, sub] of rules) r = r.replace(re, sub);
  await new Promise((res) => setTimeout(res, 150)); // fake model latency
  return r;
};
