from __future__ import annotations

from typing import Iterable, List
import difflib

import tiktoken
from ftfy import fix_text


def _clean_text(text: str) -> str:
    """Normalize text and drop unknown replacement chars.

    Using ftfy handles common encoding issues and removing the Unicode
    replacement character prevents mojibake like "�" from appearing in the
    output.
    """
    return fix_text(text).replace("\ufffd", "")


def encode_text(text: str, encoding_name: str = "cl100k_base") -> List[int]:
    """Encode text into tokens after cleaning it."""
    enc = tiktoken.get_encoding(encoding_name)
    return enc.encode(_clean_text(text))


def decode_tokens(tokens: Iterable[int], encoding_name: str = "cl100k_base") -> str:
    """Decode tokens back to text, ensuring the result is clean UTF-8."""
    enc = tiktoken.get_encoding(encoding_name)
    text = enc.decode(list(tokens))
    return _clean_text(text)


def seo_title_similarity(title: str, text: str) -> float:
    """Return a fuzzy similarity score between an SEO title and text.

    SequenceMatcher provides a ratio in [0, 1] that measures how similar two
    strings are without requiring an exact match, which is useful for
    comparing titles to content.
    """
    return difflib.SequenceMatcher(None, _clean_text(title), _clean_text(text)).ratio()

