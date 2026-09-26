"""CLIP's byte-level BPE tokenizer in plain Python + numpy, for the MobileCLIP2 text tower in ONNX.

The visual archive (core/visual_memory.py) embeds spoken queries ('a red mug') on the device, where
neither torch nor open_clip is installed. This implements the standard CLIP tokenization (byte-to-unicode
map, 49,152-merge BPE vocabulary, <start_of_text>/<end_of_text>, 77-token context, zero padding) from
the vocabulary file open_clip ships (bpe_simple_vocab_16e6.txt.gz, copied to models/ by
scripts/export_mobileclip.py). Text cleaning is html-unescape + whitespace collapse + lowercase (open_clip
also runs ftfy, which only matters for mojibake). tests/test_visual_memory.py checks the ids against
open_clip when it is importable.
"""
from __future__ import annotations

import gzip
import html
from functools import lru_cache

import numpy as np

try:
    import regex as _re                  # \p{L} / \p{N} classes, as CLIP uses
    _PAT = r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
except ImportError:                      # pragma: no cover - stdlib approximation
    import re as _re
    _PAT = r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+"""

SOT, EOT = "<start_of_text>", "<end_of_text>"


@lru_cache()
def _byte_unicode() -> dict[int, str]:
    """Bytes -> printable unicode characters (so BPE never sees whitespace or control bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipTokenizer:
    def __init__(self, vocab_path: str, context_length: int = 77):
        with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(_byte_unicode().values())
        vocab += [v + "</w>" for v in vocab] + ["".join(m) for m in merges] + [SOT, EOT]
        self.encoder = {t: i for i, t in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.context_length = context_length
        self.pat = _re.compile(_PAT, _re.IGNORECASE)
        self.cache = {SOT: SOT, EOT: EOT}
        self.sot, self.eot = self.encoder[SOT], self.encoder[EOT]

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(a, b) for a, b in zip(word, word[1:])}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        s = " ".join(word)
        self.cache[token] = s
        return s

    def encode(self, text: str) -> list[int]:
        text = " ".join(html.unescape(html.unescape(text)).split()).lower()
        bu = _byte_unicode()
        ids = []
        for tok in self.pat.findall(text):
            tok = "".join(bu[b] for b in tok.encode("utf-8"))
            ids += [self.encoder[t] for t in self._bpe(tok).split(" ")]
        return ids

    def __call__(self, texts) -> np.ndarray:
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = np.zeros((len(texts), self.context_length), np.int64)
        for i, t in enumerate(texts):
            ids = [self.sot] + self.encode(t)[: self.context_length - 2] + [self.eot]
            out[i, : len(ids)] = ids
        return out
