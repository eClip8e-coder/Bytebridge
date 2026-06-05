from __future__ import annotations

import random
import string
import unicodedata


ASCII_POOL = string.ascii_letters + string.digits + " .,;:!?-_/"


def perturb_text(text: str, perturbation_type: str, noise_level: float, seed: int) -> str:
    rng = random.Random(seed)
    if not text:
        return text
    chars = list(text)
    n_changes = max(1, int(round(len(chars) * noise_level)))

    if perturbation_type == "delete":
        for _ in range(min(n_changes, max(1, len(chars) - 1))):
            if len(chars) <= 1:
                break
            del chars[rng.randrange(len(chars))]
        return "".join(chars)

    if perturbation_type == "insert":
        for _ in range(n_changes):
            chars.insert(rng.randrange(len(chars) + 1), rng.choice(ASCII_POOL))
        return "".join(chars)

    if perturbation_type == "swap":
        for _ in range(n_changes):
            if len(chars) < 2:
                break
            i = rng.randrange(len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    if perturbation_type == "substitute":
        for _ in range(n_changes):
            chars[rng.randrange(len(chars))] = rng.choice(ASCII_POOL)
        return "".join(chars)

    if perturbation_type == "repeat":
        for _ in range(n_changes):
            i = rng.randrange(len(chars))
            chars.insert(i, chars[i])
        return "".join(chars)

    if perturbation_type == "whitespace":
        text = text.replace(" ", "  " if rng.random() < 0.5 else "\t")
        if rng.random() < noise_level:
            text = text.replace(".", " .")
        return text

    if perturbation_type == "case":
        return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in chars)

    raise ValueError(f"Unknown perturbation_type: {perturbation_type}")


def normalize_variant(text: str, form: str) -> str:
    return unicodedata.normalize(form, text)
