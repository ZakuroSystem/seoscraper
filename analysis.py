import re
from collections import Counter
from functools import lru_cache

# Simple set of English stopwords to filter common words from analysis
STOPWORDS = {
    'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'on', 'for',
    'with', 'as', 'at', 'by', 'an', 'be'
}

# Precompiled regex for tokenization to avoid recompiling on each call
_TOKEN_RE = re.compile(r"[A-Za-z]+")

@lru_cache(maxsize=128)
def analyze(text: str) -> dict:
    """Analyze text and return normalized word frequency.

    The function uses caching to avoid recomputation for repeated inputs
    and a precompiled regex for performance.
    """
    words = _TOKEN_RE.findall(text.lower())
    words = [w for w in words if w not in STOPWORDS]
    counts = Counter(words)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {w: c / total for w, c in counts.most_common()}
