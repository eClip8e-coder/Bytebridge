from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass


@dataclass
class TokenByteSpan:
    token_id: int
    token_text: str
    byte_start: int
    byte_end: int
    mapped: bool = True


def char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def token_byte_spans(text: str, tokenizer, add_special_tokens: bool = False) -> tuple[list[TokenByteSpan], dict]:
    """Map tokenizer tokens to UTF-8 byte spans.

    Fast tokenizers usually provide offset mappings. If offsets are unavailable,
    this falls back to incremental decoded-token matching. The fallback is only
    approximate and records unmapped tokens.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    byte_offsets = char_to_byte_offsets(text)
    spans: list[TokenByteSpan] = []
    unmapped = 0
    if "offset_mapping" in encoded:
        for token_id, (char_start, char_end) in zip(token_ids, encoded["offset_mapping"]):
            if char_end <= char_start:
                unmapped += 1
                spans.append(TokenByteSpan(token_id, tokenizer.decode([token_id]), 0, 0, False))
                continue
            byte_start = byte_offsets[min(char_start, len(byte_offsets) - 1)]
            byte_end = byte_offsets[min(char_end, len(byte_offsets) - 1)]
            spans.append(TokenByteSpan(token_id, tokenizer.decode([token_id]), byte_start, byte_end, True))
        return spans, {"method": "offset_mapping", "unmapped": unmapped, "total": len(token_ids)}

    cursor = 0
    raw = text.encode("utf-8")
    for token_id in token_ids:
        token_text = tokenizer.decode([token_id])
        token_bytes = token_text.encode("utf-8", errors="replace")
        pos = raw.find(token_bytes, cursor)
        if pos < 0:
            unmapped += 1
            spans.append(TokenByteSpan(token_id, token_text, cursor, cursor, False))
        else:
            spans.append(TokenByteSpan(token_id, token_text, pos, pos + len(token_bytes), True))
            cursor = pos + len(token_bytes)
    return spans, {"method": "incremental_decode", "unmapped": unmapped, "total": len(token_ids)}


def fixed_byte_spans(byte_len: int, patch_size: int, max_spans: int, max_bytes: int) -> list[tuple[int, int]]:
    spans = []
    capped = min(byte_len, max_bytes)
    for start in range(0, capped, patch_size):
        if len(spans) >= max_spans:
            break
        spans.append((start, min(start + patch_size, capped)))
    return spans


def script_class(char: str) -> str:
    if char.isspace():
        return "space"
    if char in string.punctuation:
        return "punct"
    name = unicodedata.name(char, "")
    for key in ["CJK", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "CYRILLIC"]:
        if key in name:
            return key.lower()
    if char.isdigit():
        return "digit"
    if char.isalpha():
        return "latin"
    return "symbol"


def heuristic_byte_spans(text: str, max_spans: int, max_bytes: int, max_span_bytes: int = 8) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    byte_cursor = 0
    prev_cls = None
    for char in text:
        char_bytes = len(char.encode("utf-8"))
        next_cursor = byte_cursor + char_bytes
        if byte_cursor >= max_bytes:
            break
        cls = script_class(char)
        should_break = False
        if prev_cls is not None:
            should_break = (
                cls != prev_cls
                or cls in {"space", "punct", "symbol"}
                or next_cursor - start > max_span_bytes
            )
        if should_break and byte_cursor > start:
            spans.append((start, min(byte_cursor, max_bytes)))
            start = byte_cursor
            if len(spans) >= max_spans:
                return spans
        prev_cls = cls
        byte_cursor = next_cursor
    if len(spans) < max_spans and byte_cursor > start:
        spans.append((start, min(byte_cursor, max_bytes)))
    return spans[:max_spans]


def label_spans_by_overlap(patch_spans: list[tuple[int, int]], token_spans: list[TokenByteSpan]) -> list[int]:
    labels = []
    mapped_tokens = [span for span in token_spans if span.mapped and span.byte_end > span.byte_start]
    for start, end in patch_spans:
        best_overlap = 0
        best_token = -100
        for token in mapped_tokens:
            overlap = max(0, min(end, token.byte_end) - max(start, token.byte_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_token = token.token_id
        labels.append(best_token)
    return labels
