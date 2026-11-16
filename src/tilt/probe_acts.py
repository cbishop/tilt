"""Residual-stream activations from an mlx-lm Gemma 3 student (`probe-v1`, ADR 0017).

The forward pass copies Gemma3Model.__call__ (mlx-lm 0.32.2, models/gemma3_text.py):
scaled embeddings, then the decoder blocks with the global or sliding-window mask, keeping
h after each block. Layer L = h after L blocks (1-based), before the final norm. The pass
stops after the deepest layer asked for. One sequence at a time, no batching, no cache.
"""

from __future__ import annotations

import resource
import time

import numpy as np

from tilt.probe_math import N_LAYERS, answer_span


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def chat_text(tokenizer, user: str, answer: str) -> str:
    """One user turn and one assistant turn through the chat template, no system prompt,
    no generation prompt (the source's format_chat_prompt)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False)


def span_ids(tokenizer, user: str, answer: str, exclude_last: int = 0
             ) -> tuple[list[int], int, int]:
    full = encode(tokenizer, chat_text(tokenizer, user, answer))
    empty = encode(tokenizer, chat_text(tokenizer, user, ""))
    s, e = answer_span(full, empty, exclude_last)
    return full, s, e


def memory_note() -> dict:
    out = {"maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)}
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover
        return out
    out["mlx_peak_mb"] = round(mx.get_peak_memory() / 1e6)
    return out


class Extractor:
    """One loaded student (base, or base + one LoRA adapter)."""

    def __init__(self, model_id: str, adapter_path: str | None = None):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
        self.lm = self.model.language_model.model
        assert len(self.lm.layers) == N_LAYERS, len(self.lm.layers)
        self.load_seconds = time.perf_counter() - t0
        self.passes = 0

    def _clear(self) -> None:
        self.passes += 1
        if self.passes % 200 == 0:
            clear = getattr(self.mx, "clear_cache", None) or self.mx.metal.clear_cache
            clear()

    def hidden(self, ids: list[int], upto: int) -> list:
        """[h after block 1, ..., h after block `upto`], each (T, d) in the model dtype."""
        from mlx_lm.models.base import create_attention_mask

        mx = self.mx
        lm = self.lm
        x = mx.array(ids)[None]
        h = lm.embed_tokens(x)
        h *= mx.array(lm.args.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)
        gmask = create_attention_mask(h, None)
        smask = (create_attention_mask(h, None, window_size=lm.window_size)
                 if lm.sliding_window_pattern > 1 else None)
        out = []
        for i, layer in enumerate(lm.layers[:upto]):
            is_global = i % lm.sliding_window_pattern == lm.sliding_window_pattern - 1
            h = layer(h, gmask if is_global else smask, None)
            out.append(h[0])
        return out

    def check_logits(self) -> float:
        """max |lm_head(norm(h_34)) - model(ids)| on a short prompt (should be 0)."""
        mx = self.mx
        ids = encode(self.tokenizer, chat_text(self.tokenizer, "Hi", "Hello."))
        hs = self.hidden(ids, N_LAYERS)
        lm = self.model.language_model
        normed = lm.model.norm(hs[-1][None])
        a = (lm.model.embed_tokens.as_linear(normed) if lm.tie_word_embeddings
             else lm.lm_head(normed))
        b = self.model(mx.array(ids)[None])
        return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())

    def token_rows_all_layers(self, user: str, answer: str, exclude_last: int
                              ) -> np.ndarray | None:
        """(N_LAYERS, n_tokens, d) uint16 bf16 bits of the span tokens, or None if the
        span is empty."""
        mx = self.mx
        ids, s, e = span_ids(self.tokenizer, user, answer, exclude_last)
        if e <= s:
            return None
        hs = self.hidden(ids, N_LAYERS)
        st = mx.stack([h[s:e] for h in hs])
        if st.dtype != mx.bfloat16:
            st = st.astype(mx.bfloat16)
        bits = st.view(mx.uint16)
        mx.eval(bits)
        out = np.array(bits)
        self._clear()
        return out

    def token_rows(self, user: str, answer: str, exclude_last: int, layer: int
                   ) -> np.ndarray | None:
        """(n_tokens, d) float32 span tokens at one layer, or None if the span is empty."""
        mx = self.mx
        ids, s, e = span_ids(self.tokenizer, user, answer, exclude_last)
        if e <= s:
            return None
        h = self.hidden(ids, layer)[-1][s:e].astype(mx.float32)
        mx.eval(h)
        out = np.array(h)
        self._clear()
        return out

    def pooled(self, user: str, answer: str, layers: list[int]) -> dict:
        """Mean over the answer span and the last answer token, float32, at each layer in
        `layers` (exclude_last = 0, the source's test-time span)."""
        mx = self.mx
        t0 = time.perf_counter()
        ids, s, e = span_ids(self.tokenizer, user, answer, 0)
        if e <= s:
            return {"mean": None, "last": None, "n_tokens": 0, "n_input": len(ids),
                    "seconds": time.perf_counter() - t0}
        hs = self.hidden(ids, max(layers))
        sel = [hs[L - 1] for L in layers]
        mean = mx.stack([h[s:e].astype(mx.float32).mean(axis=0) for h in sel])
        last = mx.stack([h[e - 1].astype(mx.float32) for h in sel])
        mx.eval(mean, last)
        out = {"mean": np.array(mean), "last": np.array(last), "n_tokens": e - s,
               "n_input": len(ids)}
        self._clear()
        out["seconds"] = time.perf_counter() - t0
        return out
