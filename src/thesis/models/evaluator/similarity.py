"""Utility Evaluator (D-016): semantic similarity via all-mpnet-base-v2.

S_sem(original, rewrite) = cosine similarity of normalized sentence embeddings,
the utility term in the DPO reward R = lambda*S_sem - (1-lambda)*P_att. Off-the-shelf,
not retrained. Small (110M) and fast — it is called for every candidate rewrite.
"""

from __future__ import annotations


class SemanticSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def cosine_batch(self, original: str, candidates: list[str]) -> list[float]:
        """Cosine of `original` against each candidate (normalized → dot product)."""
        if not candidates:
            return []
        emb = self.model.encode([original, *candidates], convert_to_tensor=True,
                                normalize_embeddings=True)
        o = emb[0]
        return [float((o * emb[i + 1]).sum().item()) for i in range(len(candidates))]

    def cosine(self, a: str, b: str) -> float:
        return self.cosine_batch(a, [b])[0]
