import argparse
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import List, Optional, Tuple, Dict, Set, Iterable
from collections import defaultdict, Counter
import re
import json
import csv
import os
from functools import lru_cache
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from googlesearch import search
import tldextract
import tiktoken
from janome.tokenizer import Tokenizer

OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


# =========================
# ロギング
# =========================
def setup_logging(log_file: Optional[str], level: str, max_bytes: int, backup_count: int):
    """コンソール + （任意）ファイルへログ出力。ファイルはローテーション付き。"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(fh)


# =========================
# HTTP / 検索
# =========================
def create_session() -> requests.Session:
    """Return a requests session with retry and user-agent."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def get_search_results(query: str, num_results: int, pause: float) -> List[str]:
    """Return a list of URLs from Google search."""
    try:
        urls = list(search(query, num_results=num_results, sleep_interval=pause))
        logging.info("Search ok: %s (hits=%d)", query, len(urls))
        return urls
    except Exception as e:
        logging.error("Search failed: %s", e)
        return []


# =========================
# 文字化け対策（デコード最適化＋スキップ）
# =========================
_MOJIBAKE_PATTERNS = [
    r"Ã.", r"Â.", r"â..", r"ðŸ", r"�",   # ラテン系崩れ/置換文字
    r"ã‚", r"ãƒ", r"ã„", r"ãŒ",          # UTF-8→SJIS/EUC 誤読
    r"å.", r"æ.", r"œ"
]
_MOJIBAKE_REGEX = re.compile("|".join(_MOJIBAKE_PATTERNS))

def mojibake_score(text: str) -> float:
    """ざっくり文字化けスコア（0に近いほど正常）"""
    if not text:
        return 1.0
    hits = len(_MOJIBAKE_REGEX.findall(text))
    hits += text.count("�") * 2
    hits += len(re.findall(r"(Ã|Â|â){3,}", text)) * 3
    return hits / max(len(text), 1)


def _find_meta_charset(head_bytes: bytes) -> Optional[str]:
    """<meta charset=...> / http-equiv を先頭2KBから拾う"""
    head = head_bytes.decode("latin-1", errors="ignore")
    m = re.search(r'<meta[^>]+charset=["\']?\s*([\w\-:]+)\s*', head, flags=re.I)
    if m:
        return m.group(1).lower()
    m = re.search(
        r'<meta[^>]+http-equiv=["\']content-type["\'][^>]*content=["\'][^">]*charset=([\w\-:]+)',
        head, flags=re.I)
    if m:
        return m.group(1).lower()
    return None


def _unique_clean(seq: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if not s:
            continue
        s = s.strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def best_decode(response: requests.Response) -> Tuple[str, str, float]:
    """複数エンコーディングで試し、最も文字化けスコアが低いテキストを返す。"""
    raw = response.content
    head = raw[:2048]
    candidates = _unique_clean([
        "utf-8",
        response.encoding,
        getattr(response, "apparent_encoding", None),
        _find_meta_charset(head),
        # 日本語でよく使われるもの
        "cp932", "shift_jis", "euc-jp", "iso-2022-jp",
        # 欧文系
        "windows-1252", "latin-1"
    ])
    best_txt, best_enc, best_score = "", candidates[0] if candidates else "utf-8", 1e9
    for enc in candidates or ["utf-8"]:
        try:
            txt = raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            txt = raw.decode(enc, errors="replace")
        score = mojibake_score(txt[:20000])
        logging.debug("Try decode enc=%s score=%.5f url=%s", enc, score, response.url)
        if score < best_score:
            best_txt, best_enc, best_score = txt, enc, score
        if best_score < 0.0015:
            break
    logging.info("Decoded with enc=%s score=%.5f url=%s", best_enc, best_score, response.url)
    return best_txt, best_enc, best_score


def fetch_html(session: requests.Session, url: str, timeout: int = 10) -> Optional[str]:
    """複数エンコーディングで再デコードし、文字化けなら None を返す。"""
    try:
        logging.debug("GET %s", url)
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        text, used_enc, score = best_decode(resp)
        if score >= 0.008:
            logging.warning("Skip (mojibake likely) score=%.4f enc=%s url=%s", score, used_enc, url)
            return None
        return text
    except Exception as e:
        logging.warning("Failed to fetch %s: %s", url, e)
        return None


# =========================
# HTML解析
# =========================
def robots_exists(session: requests.Session, url: str) -> bool:
    """Return True if robots.txt exists for the URL's domain."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = session.get(robots_url, timeout=5)
        ok = resp.status_code == 200
        logging.debug("robots.txt %s -> %s", robots_url, "OK" if ok else resp.status_code)
        return ok
    except Exception as e:
        logging.info("robots.txt fetch failed for %s: %s", robots_url, e)
        return False


def parse_html(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    text = ' '.join(p.get_text(separator=' ', strip=True) for p in soup.find_all('p'))
    published_time = None
    seo_title = None
    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f'h{level}'):
            content = h.get_text(strip=True)
            if content:
                headings.append(content)

    time_selectors = [
        ('meta', {'property': 'article:published_time'}),
        ('meta', {'property': 'og:published_time'}),
        ('meta', {'name': 'pubdate'}),
        ('meta', {'name': 'publish-date'}),
        ('meta', {'name': 'date'})
    ]
    for tag, attrs in time_selectors:
        el = soup.find(tag, attrs=attrs)
        if el and el.get('content'):
            published_time = el['content']
            break
    if not published_time:
        el = soup.find('time')
        if el and el.get('datetime'):
            published_time = el['datetime']
    if not published_time:
        published_time = 'N/A'

    title_selectors = [
        ('meta', {'property': 'og:title'}),
        ('meta', {'name': 'title'}),
        ('meta', {'name': 'twitter:title'})
    ]
    for tag, attrs in title_selectors:
        el = soup.find(tag, attrs=attrs)
        if el and el.get('content'):
            seo_title = el['content']
            break
    if not seo_title and soup.title:
        seo_title = soup.title.get_text(strip=True)
    if not seo_title:
        seo_title = 'N/A'

    desc_selectors = [
        ('meta', {'name': 'description'}),
        ('meta', {'property': 'og:description'}),
        ('meta', {'name': 'twitter:description'}),
    ]
    description = ''
    for tag, attrs in desc_selectors:
        el = soup.find(tag, attrs=attrs)
        if el and el.get('content'):
            description = el['content']
            break
    image_count = len(soup.find_all('img'))
    link_count = len(soup.find_all('a'))

    logging.debug(
        "Parsed title=%s published=%s desc_len=%d images=%d links=%d",
        seo_title,
        published_time,
        len(description),
        image_count,
        link_count,
    )
    return {
        'text': text,
        'published_time': published_time,
        'title': seo_title,
        'description': description,
        'images': image_count,
        'links': link_count,
        'headings': headings,
    }


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return '.'.join(part for part in [ext.domain, ext.suffix] if part)


# =========================
# テキスト解析：語数・キーワード頻度
# =========================
def _insertion_cost(n: int) -> float:
    if n == 1:
        return 1.0
    if n == 2:
        return 1.5
    return 3.0


def _edit_distance(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    dp = [[(0.0, 0) for _ in range(lb + 1)] for _ in range(la + 1)]
    for i in range(1, la + 1):
        cost, ins = dp[i - 1][0]
        dp[i][0] = (cost + 2.0, ins)
    for j in range(1, lb + 1):
        cost, ins = dp[0][j - 1]
        new_ins = ins + 1
        dp[0][j] = (cost + _insertion_cost(new_ins), new_ins)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            # deletion
            del_cost, del_ins = dp[i - 1][j]
            del_cost += 2.0
            # insertion
            ins_cost, ins_ins = dp[i][j - 1]
            new_ins = ins_ins + 1
            ins_cost += _insertion_cost(new_ins)
            ins_ins = new_ins
            # substitution / match
            sub_cost, sub_ins = dp[i - 1][j - 1]
            if a[i - 1] != b[j - 1]:
                sub_cost += 3.0
            candidates = [
                (del_cost, del_ins),
                (ins_cost, ins_ins),
                (sub_cost, sub_ins),
            ]
            dp[i][j] = min(candidates, key=lambda x: x[0])
    return dp[la][lb][0]


def _merge_similar(counter: Counter, threshold: float) -> Counter:
    merged: Dict[str, int] = {}
    for token, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        for canon in list(merged.keys()):
            dist = _edit_distance(token, canon)
            norm = max(len(token), len(canon)) * 1.5
            if dist / norm <= threshold:
                target = token if len(token) > len(canon) else canon
                merged[target] = merged.pop(canon) + cnt
                break
        else:
            merged[token] = cnt
    return Counter(merged)


def analyze_keywords(text: str, top_n: int = 10, merge_threshold: float = 0.0) -> Tuple[int, List[Dict[str, int]]]:
    local = analyze_keywords._local
    tokenizer = getattr(local, "tokenizer", None)
    if tokenizer is None:
        tokenizer = Tokenizer()
        local.tokenizer = tokenizer
    counter: Counter = Counter()
    # Janome's tokenizer and underlying FST are not thread‑safe, so guard usage
    # with a global lock. If a KeyError bubbles up from the dictionary cache,
    # recreate the tokenizer and retry once to avoid crashing the worker.
    with analyze_keywords._lock:
        try:
            tokens = list(tokenizer.tokenize(text))
        except KeyError:
            tokenizer = Tokenizer()
            local.tokenizer = tokenizer
            tokens = list(tokenizer.tokenize(text))
    for t in tokens:
        pos = t.part_of_speech.split(',')[0]
        base = t.base_form if t.base_form != '*' else t.surface
        if pos == '名詞' and base not in analyze_keywords._stopwords and len(base) > 1:
            counter[base] += 1
    if merge_threshold > 0:
        counter = _merge_similar(counter, merge_threshold)
    total = sum(counter.values())
    top = [{"keyword": k, "count": c} for k, c in counter.most_common(top_n)]
    return total, top

analyze_keywords._local = threading.local()
analyze_keywords._lock = threading.Lock()
analyze_keywords._stopwords = {
    'する', 'ます', 'ある', 'いる', 'なる', 'こと', 'これ', 'それ', 'さん'
}

# =========================
# 除外定義（文字/正規表現）の読み込み
# =========================
def parse_exclude_lines(lines: Iterable[str]) -> Tuple[Set[str], Optional[Dict[int, None]], List[re.Pattern]]:
    chars: Set[str] = set()
    patterns: List[re.Pattern] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('/') and line.endswith('/') and len(line) >= 2:
            try:
                patterns.append(re.compile(line[1:-1]))
            except re.error as e:
                logging.warning("Invalid regex %r: %s", line, e)
        else:
            for ch in line:
                if ch not in ("\n", "\r"):
                    chars.add(ch)
    delete_table = str.maketrans("", "", "".join(sorted(chars))) if chars else None
    return chars, delete_table, patterns


def load_excludes(path: Optional[str]) -> Tuple[Set[str], Optional[Dict[int, None]], List[re.Pattern]]:
    """
    UTF-8 (BOM可) のテキストファイルから除外文字と正規表現を読み込む。
    各行に文字列を記述するとその各文字を除去対象にし、`/regex/` 形式の行は
    正規表現として扱う。
    """
    if not path:
        return set(), None, []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        chars, delete_table, patterns = parse_exclude_lines(lines)
        logging.info("Loaded excludes: chars=%d regex=%d from %s", len(chars), len(patterns), path)
        return chars, delete_table, patterns
    except Exception as e:
        logging.error("Failed to load exclude file %s: %s", path, e)
        return set(), None, []


# =========================
# 共通判定（文字列/トークン単位, 3～12長・最長一致優先）
# =========================
_PRESENT_CHAR = re.compile(r'[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]')

def _normalize_for_substrings(
    s: str,
    remove_trans: Optional[Dict[int, None]] = None,
    remove_patterns: Optional[List[re.Pattern]] = None,
) -> str:
    """除外文字/正規表現の削除 → 連続空白圧縮 → 前後空白除去。"""
    if not s:
        return ""
    if remove_trans:
        s = s.translate(remove_trans)
    if remove_patterns:
        for pat in remove_patterns:
            s = pat.sub(' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

_NOISE_PHRASES = {
    "です", "です。", "ます", "ます。", "します", "します。",
    "しています", "してい", "ている", "ください", "あります", "ありま",
    "サービス", "事業所",
}
_KANA_ONLY = re.compile(r'^[\u3040-\u30FF]+$')

def filter_common_phrases(items: List[Tuple[str, int]], keyword: str) -> List[Tuple[str, int]]:
    """共通結果からノイズと検索語を除外"""
    norm_kw = _normalize_for_substrings(keyword)
    parts = [p for p in re.split(r'\s+', norm_kw) if p]
    noise = set(parts) | _NOISE_PHRASES
    filtered: List[Tuple[str, int]] = []
    for sub, cnt in items:
        if sub != sub.strip():
            continue
        s = sub.strip()
        if not s or s in noise or s in norm_kw:
            continue
        if _KANA_ONLY.fullmatch(s) and len(s) <= 4:
            continue
        filtered.append((sub, cnt))
    return filtered

def _iter_substrings(s: str, min_len: int, max_len: int) -> Iterable[str]:
    """長さ制約内の部分文字列をすべて生成（文字列そのまま、トークナイザ不使用）。"""
    n = len(s)
    max_len = min(max_len, n)
    for L in range(min_len, max_len + 1):
        for i in range(0, n - L + 1):
            sub = s[i:i+L]
            # 文字種フィルタ：完全な空白/記号列・同一文字の繰り返しのみは除外
            if _PRESENT_CHAR.search(sub) and len(set(sub)) > 1:
                yield sub

def common_substrings_rank_from_norm(
    norm_texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    top_k: int = 15,
    max_doc_ratio: float = 0.8
) -> List[Tuple[str, int]]:
    """
    すでに正規化済み（除外適用＆空白圧縮済み＆必要なら切り詰め済み）の本文配列から
    共通部分文字列を抽出（最長一致優先）。
    max_doc_ratio を超える頻出部分文字列は汎用的とみなして除外する。
    """
    sub_to_docs: Dict[str, Set[int]] = defaultdict(set)
    total_docs = len(norm_texts)
    for doc_id, s in enumerate(norm_texts):
        if not s:
            continue
        seen_in_doc: Set[str] = set()
        for sub in _iter_substrings(s, min_len, max_len):
            if sub in seen_in_doc:
                continue
            seen_in_doc.add(sub)
            sub_to_docs[sub].add(doc_id)

    grouped: Dict[frozenset, List[str]] = defaultdict(list)
    for sub, docs in sub_to_docs.items():
        if len(docs) >= 2 and len(docs) / total_docs <= max_doc_ratio:
            grouped[frozenset(docs)].append(sub)

    filtered: List[Tuple[str, int]] = []
    for docset, subs in grouped.items():
        subs.sort(key=lambda x: (-len(x), x))  # 長い順 → 同長は辞書順
        kept: List[str] = []
        for sub in subs:
            if any(ks.find(sub) != -1 for ks in kept):
                continue
            kept.append(sub)
        for sub in kept:
            filtered.append((sub, len(docset)))

    filtered.sort(key=lambda t: (-t[1], -len(t[0]), t[0]))
    return filtered[:top_k]


def _iter_token_sequences(tokens: List[int], min_len: int, max_len: int) -> Iterable[Tuple[int, ...]]:
    """長さ制約内のトークン列をすべて生成。"""
    n = len(tokens)
    max_len = min(max_len, n)
    for L in range(min_len, max_len + 1):
        for i in range(0, n - L + 1):
            seq = tuple(tokens[i:i+L])
            if len(set(seq)) > 1:
                yield seq


def _token_seq_contains(seq: Tuple[int, ...], sub: Tuple[int, ...]) -> bool:
    """seq が sub を部分列として含むか判定。"""
    n, m = len(seq), len(sub)
    if m > n:
        return False
    for i in range(n - m + 1):
        if seq[i:i+m] == sub:
            return True
    return False


def common_tokens_rank_from_norm(
    norm_texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    top_k: int = 15,
    max_doc_ratio: float = 0.8,
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """
    正規化済み本文配列から共通トークン列を抽出。tiktoken でトークン化し、
    max_doc_ratio を超える頻出トークン列は除外する。
    """
    enc = tiktoken.get_encoding(encoding_name)
    token_texts = [enc.encode(t) for t in norm_texts]
    sub_to_docs: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
    total_docs = len(token_texts)
    for doc_id, tokens in enumerate(token_texts):
        if not tokens:
            continue
        seen_in_doc: Set[Tuple[int, ...]] = set()
        for seq in _iter_token_sequences(tokens, min_len, max_len):
            if seq in seen_in_doc:
                continue
            seen_in_doc.add(seq)
            sub_to_docs[seq].add(doc_id)

    grouped: Dict[frozenset, List[Tuple[int, ...]]] = defaultdict(list)
    for seq, docs in sub_to_docs.items():
        if len(docs) >= 2 and len(docs) / total_docs <= max_doc_ratio:
            grouped[frozenset(docs)].append(seq)

    filtered: List[Tuple[Tuple[int, ...], int]] = []
    for docset, seqs in grouped.items():
        seqs.sort(key=lambda x: (-len(x), x))
        kept: List[Tuple[int, ...]] = []
        for seq in seqs:
            if any(_token_seq_contains(k, seq) for k in kept):
                continue
            kept.append(seq)
        for seq in kept:
            filtered.append((seq, len(docset)))

    filtered.sort(key=lambda t: (-t[1], -len(t[0]), t[0]))
    results: List[Tuple[str, int]] = []
    for seq, cnt in filtered:
        try:
            text = enc.decode(list(seq), errors="strict")
        except UnicodeDecodeError:
            continue
        if "\ufffd" in text:
            continue
        results.append((text, cnt))
        if len(results) >= top_k:
            break
    return results


def hybrid_common_rank_from_norm(
    norm_texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    top_k: int = 15,
    max_doc_ratio: float = 0.8,
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """Tokenと文字列の両方で共通判定し一致したもののみ返す。"""
    token_ranks = common_tokens_rank_from_norm(
        norm_texts,
        min_len=min_len,
        max_len=max_len,
        top_k=top_k,
        max_doc_ratio=max_doc_ratio,
        encoding_name=encoding_name,
    )
    char_ranks = common_substrings_rank_from_norm(
        norm_texts,
        min_len=min_len,
        max_len=max_len,
        top_k=top_k,
        max_doc_ratio=max_doc_ratio,
    )
    char_map = {s: c for s, c in char_ranks}
    hybrid = [(s, min(cnt, char_map[s])) for s, cnt in token_ranks if s in char_map]
    hybrid.sort(key=lambda t: (-t[1], -len(t[0]), t[0]))
    return hybrid[:top_k]

def rank_common_titles_from_norm(
    norm_titles: List[str],
    top_k: int = 15,
    min_len: int = 3,
    max_len: int = 12,
    max_doc_ratio: float = 0.8,
    mode: str = "tiktoken",
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """正規化済みタイトルの共通部分列ランキング。"""
    if mode == "char":
        return common_substrings_rank_from_norm(
            norm_titles,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
        )
    elif mode == "hybrid":
        return hybrid_common_rank_from_norm(
            norm_titles,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
            encoding_name=encoding_name,
        )
    else:
        return common_tokens_rank_from_norm(
            norm_titles,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
            encoding_name=encoding_name,
        )

def common_substrings_rank(
    texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    analyze_chars: int = 5000,
    top_k: int = 15,
    remove_trans: Optional[Dict[int, None]] = None,
    remove_patterns: Optional[List[re.Pattern]] = None,
    max_doc_ratio: float = 0.8,
    mode: str = "tiktoken",
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """（オンライン計算用）正規化→切り詰め→共通部分列抽出。
    mode に応じて文字列 or トークン列で判定。"""
    norm_texts = [
        _normalize_for_substrings(t, remove_trans, remove_patterns)[:analyze_chars]
        for t in texts
    ]
    if mode == "char":
        return common_substrings_rank_from_norm(
            norm_texts,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
        )
    elif mode == "hybrid":
        return hybrid_common_rank_from_norm(
            norm_texts,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
            encoding_name=encoding_name,
        )
    else:
        return common_tokens_rank_from_norm(
            norm_texts,
            min_len=min_len,
            max_len=max_len,
            top_k=top_k,
            max_doc_ratio=max_doc_ratio,
            encoding_name=encoding_name,
        )

def rank_common_titles(
    titles: List[str],
    top_k: int = 15,
    remove_trans: Optional[Dict[int, None]] = None,
    remove_patterns: Optional[List[re.Pattern]] = None,
    mode: str = "tiktoken",
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """（オンライン計算用）正規化→共通部分列ランキング。"""
    normed = []
    for t in titles:
        if not t:
            continue
        s = t
        if remove_trans:
            s = s.translate(remove_trans)
        if remove_patterns:
            for pat in remove_patterns:
                s = pat.sub(' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            normed.append(s)
    return rank_common_titles_from_norm(
        normed,
        top_k=top_k,
        mode=mode,
        encoding_name=encoding_name,
    )



# =========================
# AIによるブログ指示書生成 (Ollama)
# =========================


@lru_cache()
def have_ollama_model(name: str) -> bool:
    """Return True if the given Ollama model exists locally."""
    try:
        import requests

        resp = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
        data = resp.json()
        return any(m.get("name") == name for m in data.get("models", []))
    except Exception as e:  # pragma: no cover - optional runtime feature
        logging.info("Ollama model check failed: %s", e)
        return False


def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 160,
    web_search: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Send a chat request to the Ollama server and return the response text.

    Before the main request, a small "Hello" handshake is performed (60s timeout)
    to ensure the backend model is responsive. Returns a tuple of
    (content, error_message). On success, error_message is None.
    """
    try:
        import json
        import requests

        # handshake
        web_flag = web_search or os.environ.get("OLLAMA_WEB_SEARCH") == "1"
        hello_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if web_flag:
            hello_payload["web_search"] = True
        hello_resp = requests.post(
            f"{OLLAMA_API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(hello_payload),
            timeout=60,
        )
        hello_data = hello_resp.json()
        if hello_resp.status_code != 200 or "error" in hello_data:
            raise RuntimeError(hello_data.get("error", hello_resp.text))

        payload = {"model": model, "messages": messages}
        if web_flag:
            payload["web_search"] = True
        resp = requests.post(
            f"{OLLAMA_API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        data = resp.json()
        if resp.status_code != 200 or "error" in data:
            raise RuntimeError(data.get("error", resp.text))
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("no choices in response")
        content = choices[0].get("message", {}).get("content", "").strip()
        return content, None
    except Exception as e:
        logging.info("Ollama chat failed: %s", e)
        return None, str(e)


def generate_blog_instruction(
    keyword: str,
    results: List[Dict],
    common_subs: List[Dict],
    title_ranks: List[Tuple[str, int]],
    user_prompt: str = "",
    info_only: bool = False,
    model: str = "gpt-oss:20b",
    timeout: int = 160,
    web_search: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """検索結果の概要から SEO ブログ記事の指示書を生成する。"""
    if not have_ollama_model(model):
        msg = f"{model} not available"
        logging.info(msg)
        return None, msg
    summary_lines = []
    for r in results[:5]:
        kws = ", ".join(k["keyword"] for k in r.get("top_keywords", [])[:3])
        heads = "/".join(r.get("headings", [])[:3])
        summary_lines.append(f"- {r.get('title', '')} | 見出し: {heads} | キーワード: {kws}")
    body = "\n".join(summary_lines)
    subs = "\n".join(f"- {s['text']} ({s['count']}件)" for s in common_subs[:5])
    titles = "\n".join(f"- {t} ({c}件)" for t, c in title_ranks[:5])
    prompt = (
        f"検索キーワード: {keyword}\n"
        f"上位ページの概要:\n{body}\n\n"
    )
    if subs:
        prompt += f"共通本文フレーズ:\n{subs}\n\n"
    prompt += (
        f"共通SEOタイトルフレーズ:\n{titles}\n\n"
        "これらを参考に、どのようなブログ記事を書けばよいかを日本語でまとめた指示書を作成してください。"
    )
    if info_only:
        prompt += "案件や見積りへの誘導は避け、純粋な情報提供に徹してください。"
    if user_prompt:
        prompt += f"\n\n追加指示:\n{user_prompt}"
    messages = [
        {"role": "system", "content": "You are an expert Japanese SEO consultant."},
        {"role": "user", "content": prompt},
    ]
    return ollama_chat(model, messages, timeout=timeout, web_search=web_search)


def generate_blog_post(
    keyword: str,
    instructions: str,
    user_prompt: str = "",
    style: str = "",
    human_mode: bool = False,
    info_only: bool = False,
    html_mode: bool = False,
    model: str = "gpt-oss:20b",
    timeout: int = 160,
    web_search: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """ブログ指示書から記事本文を生成する。"""
    if not have_ollama_model(model):
        msg = f"{model} not available"
        logging.info(msg)
        return None, msg
    fmt = "HTML" if html_mode else "Markdown"
    prompt = (
        f"検索キーワード: {keyword}\n"
        f"以下の指示書に従って、日本語でSEOに最適化されたブログ記事を{fmt}形式で作成してください。\n\n"
        f"{instructions}\n"
    )
    if style:
        prompt += f"\nブログの雰囲気: {style}\n"
    if human_mode:
        prompt += (
            "\nAIで生成されたと気付かれないよう、人間らしい自然な語り口で書いてください。"
            "以下の点にも留意してください:\n"
            "- 見出し番号など機械的な構成を避け、節ごとに変化を付ける\n"
            "- 金額や数値は現実的な揺らぎを含め、過度に整列させない\n"
            "- 不自然な語句や誤用を避け、文脈に合う語彙を選ぶ\n"
            "- 汎用的なマーケティング文句を多用せず具体例や根拠を示す\n"
            "- 表やFAQなどの装飾は必要なものだけに留める\n"
        )
    if user_prompt:
        prompt += f"\n追加指示:\n{user_prompt}\n"
    if info_only:
        prompt += "\n案件や見積りへの誘導は行わず、読者への情報提供のみに集中してください。"
    messages = [
        {
            "role": "system",
            "content": f"You are a skilled Japanese blogger. Output {fmt}.",
        },
        {"role": "user", "content": prompt},
    ]
    return ollama_chat(model, messages, timeout=timeout, web_search=web_search)


def ollama_chat_stream(
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 160,
    web_search: bool = False,
):
    """Yield content chunks from Ollama as they arrive."""
    import json
    import requests

    # handshake
    web_flag = web_search or os.environ.get("OLLAMA_WEB_SEARCH") == "1"
    hello_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if web_flag:
        hello_payload["web_search"] = True
    requests.post(
        f"{OLLAMA_API_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(hello_payload),
        timeout=60,
    )

    payload = {"model": model, "messages": messages, "stream": True}
    if web_flag:
        payload["web_search"] = True
    with requests.post(
        f"{OLLAMA_API_BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                payload = line[6:]
                if payload.strip() == b"[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    logging.info("Bad stream line: %r", line)
                    continue
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

def save_markdown(content: str, keyword: str, directory: str = ".", ext: str = "md") -> str:
    """テキストファイルを保存し、保存先パスを返す。"""
    os.makedirs(directory, exist_ok=True)
    safe_kw = re.sub(r"[^0-9A-Za-z_-]+", "_", keyword)[:30]
    filename = f"{safe_kw}_{int(time.time())}.{ext}"
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info("file saved: %s", path)
    return path


def save_blog_markdown(content: str, keyword: str, directory: str = ".", html: bool = False) -> str:
    """ブログ記事をファイルとして保存し、パスを返す。"""
    ext = "html" if html else "md"
    return save_markdown(content, keyword, directory, ext=ext)


def save_report_markdown(content: str, keyword: str, directory: str = ".") -> str:
    """レポートをMarkdownファイルとして保存し、パスを返す。"""
    return save_markdown(content, keyword, directory, ext="md")


def save_scrape_json(results: List[Dict], keyword: str, directory: str = ".") -> str:
    """スクレイピング結果をJSONとして保存し、保存先パスを返す。"""
    os.makedirs(directory, exist_ok=True)
    safe_kw = re.sub(r"[^0-9A-Za-z_-]+", "_", keyword)[:30]
    filename = f"{safe_kw}_{int(time.time())}.json"
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Scrape JSON saved: %s", path)
    return path


# =========================
# 結果・分析ファイルの書き出し／読み込み
# =========================
def write_results_csv(path: str, rows: List[Dict]):
    fieldnames = [
        "url",
        "domain",
        "published_time",
        "title",
        "description",
        "robots",
        "word_count",
        "top_keywords",
        "images",
        "links",
        "headings",
        "text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r_copy = r.copy()
            if isinstance(r_copy.get("top_keywords"), list):
                r_copy["top_keywords"] = ";".join(f"{k['keyword']}:{k['count']}" for k in r_copy['top_keywords'])
            if isinstance(r_copy.get("headings"), list):
                r_copy["headings"] = ";".join(r_copy["headings"])
            writer.writerow(r_copy)
    logging.info("Results CSV written: %s", path)


def write_results_json(path: str, rows: List[Dict], meta: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": rows}, f, ensure_ascii=False, indent=2)
    logging.info("Results JSON written: %s", path)


def save_analysis(
    path: str,
    meta: Dict,
    norm_texts: List[str],
    norm_titles: List[str],
    results_rows: List[Dict]
):
    """
    再集計専用の“分析ファイル”を保存。
    - norm_texts: 除外適用・空白正規化・analyze_charsで切り詰め済みの本文
    - norm_titles: 除外適用・空白正規化済みタイトル
    - results_rows: URL/タイトル等の表示用
    """
    data = {
        "version": 1,
        "meta": meta,
        "norm_texts": norm_texts,
        "norm_titles": norm_titles,
        "results": results_rows
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info("Analysis file written: %s", path)


def load_analysis(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "version" not in data:
        data["version"] = 1
    logging.info("Analysis file loaded: %s (version=%s)", path, data.get("version"))
    return data


# =========================
# メイン
# =========================
def main():
    parser = argparse.ArgumentParser(description='ウェブを検索し情報を抽出するツール')
    parser.add_argument('keyword', nargs='?', help='検索キーワード（--analysis-load を使う場合は省略可）')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='取得するURLの件数')
    parser.add_argument('--delay', type=float, default=0.0, help='各リクエスト前の待機秒数')
    parser.add_argument('--workers', type=int, default=10, help='同時リクエスト数')
    parser.add_argument('--chars', type=int, default=1000, help='本文の表示文字数')
    parser.add_argument('--merge-percent', type=float, default=18.0,
                        help='類似キーワードを統合する最大編集距離(%)')
    # ログ
    parser.add_argument('--log-file', default=None, help='ログ出力先ファイル（指定しない場合はコンソールのみ）')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='ログレベル')
    parser.add_argument('--log-max-bytes', type=int, default=5*1024*1024, help='ローテーション閾値（バイト）')
    parser.add_argument('--log-backup-count', type=int, default=3, help='ローテーション世代数')
    # 出力
    parser.add_argument('--results-csv', default=None, help='結果CSVのパス')
    parser.add_argument('--results-json', default=None, help='結果JSONのパス')
    parser.add_argument('--show-pages', action='store_true', help='ページ別の生データを表示する')
    # 共通判定パラメータ
    parser.add_argument('--rank-k', type=int, default=15, help='ランキングの表示件数（共通本文サブ文字列 / 共通SEOタイトルサブ文字列）')
    parser.add_argument('--analyze-chars', type=int, default=5000, help='共通判定に用いる本文の先頭文字数（デフォルト5000）')
    parser.add_argument('--exclude-file', default=None, help='除外文字や正規表現の一覧ファイル。`/regex/` 形式の行は正規表現として扱う。')
    parser.add_argument('--max-common-ratio', type=float, default=0.8,
                        help='共通サブ文字列として扱う最大出現率（0.0-1.0、デフォルト0.8）')
    parser.add_argument('--analysis-mode', choices=['tiktoken', 'char', 'hybrid'], default='tiktoken',
                        help='解析モード: tiktoken / char / hybrid')
    # 分析ファイル
    parser.add_argument('--analysis-save', default=None, help='正規化済みテキスト等を保存する分析ファイル(JSON)のパス')
    parser.add_argument('--analysis-load', default=None, help='分析ファイル(JSON)を読み込みローカル再集計のみ行う（ネットワークアクセス無し）')
    # 互換（旧オプション）: --top-k が与えられたら --rank-k に流用
    parser.add_argument('--top-k', type=int, default=None, help='[互換] ランキング件数。指定時は --rank-k を上書き')
    args = parser.parse_args()

    if args.top_k is not None:
        args.rank_k = args.top_k

    setup_logging(args.log_file, args.log_level, args.log_max_bytes, args.log_backup_count)

    # ============ 分析ファイルのロードモード（ローカル再集計） ============
    if args.analysis_load:
        data = load_analysis(args.analysis_load)

        # 互換チェック（除外文字や analyze_chars が違う場合は警告。top-kだけ変更なら問題なし）
        saved_meta = data.get("meta", {})
        saved_excl = set(saved_meta.get("exclude_chars", []))
        saved_regex = saved_meta.get("exclude_regex", [])
        saved_analyze_chars = saved_meta.get("analyze_chars")
        saved_mode = saved_meta.get("analysis_mode")
        # 現在の除外ファイルを読み込んだ場合は一致比較
        exclude_chars, remove_trans, exclude_regex = load_excludes(args.exclude_file)
        if exclude_chars and exclude_chars != saved_excl:
            logging.warning("Exclude chars differ from analysis file. Recalc uses SAVED normalization.")
        if exclude_regex and [p.pattern for p in exclude_regex] != saved_regex:
            logging.warning("Exclude regex differ from analysis file. Recalc uses SAVED normalization.")
        if args.analyze_chars and saved_analyze_chars and args.analyze_chars != saved_analyze_chars:
            logging.warning("analyze_chars differs from analysis file. Recalc uses SAVED truncation.")
        if saved_mode and args.analysis_mode != saved_mode:
            logging.warning("analysis_mode differs from analysis file. Recalc uses %s", args.analysis_mode)

        norm_texts = data.get("norm_texts", [])
        norm_titles = data.get("norm_titles", [])
        results = data.get("results", [])

        # ローカル再集計（top-k 変更だけなら超高速）
        if args.analysis_mode == 'char':
            raw_subs = common_substrings_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        elif args.analysis_mode == 'hybrid':
            raw_subs = hybrid_common_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        else:
            raw_subs = common_tokens_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        filtered = filter_common_phrases(raw_subs, saved_meta.get("keyword", ""))[:args.rank_k]
        total_docs = len(norm_texts) if norm_texts else 1
        common_subs = [
            {"text": sub, "count": cnt, "ratio": cnt / total_docs}
            for sub, cnt in filtered
        ]
        title_ranks = rank_common_titles_from_norm(
            norm_titles,
            top_k=args.rank_k,
            mode=args.analysis_mode,
        )

        logging.info("Re-aggregated locally from analysis file. rank_k=%d", args.rank_k)

        # 結果JSONを書きたい場合は新たに出力
        if args.results_json:
            meta = {
                "source": "analysis_load",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rank_k": args.rank_k,
                "keyword": saved_meta.get("keyword"),
                "analyze_chars": saved_meta.get("analyze_chars"),
                "max_common_ratio": args.max_common_ratio,
                "analysis_mode": args.analysis_mode,
                "merge_percent": args.merge_percent,
                "exclude_chars_count": len(saved_excl),
                "exclude_regex": saved_regex,
                "exclude_regex_count": len(saved_regex),
                "common_substrings": common_subs,
                "common_seo_title_substrings": title_ranks,
            }
            write_results_json(args.results_json, results, meta)

        # 従来の出力（--show-pages 指定時のみ）
        if args.show_pages:
            for item in results:
                print("URL:", item['url'])
                print("ドメイン:", item['domain'])
                print("公開日:　", item['published_time'])
                print("SEOタイトル:", item['title'])
                print("説明:", item.get('description', ''))
                print("robots.txt:　", "あり" if item['robots'] else "なし")
                print("語数:", item['word_count'])
                print("上位キーワード:", ', '.join(f"{k['keyword']}:{k['count']}" for k in item['top_keywords']))
                print("画像数:", item.get('images'))
                print("リンク数:", item.get('links'))
                print("本文:　", item['text'])
                print("-" * 80)

        unit = "文字" if args.analysis_mode == 'char' else "トークン"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        print(f"共通本文サブ文字列（3～12{unit}, 最長一致・上位{args.rank_k}）:")
        if common_subs:
            for item in common_subs:
                sub = item["text"]
                cnt = item["count"]
                ratio = item["ratio"]
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                print(f"[{cnt}件 / {length}{unit} / {ratio:.0%}] {repr(sub)}")
        else:
            print("（該当なし）")

        print(f"共通SEOタイトルサブ文字列（3～12{unit}、上位{args.rank_k}）:")
        if title_ranks:
            for title, cnt in title_ranks:
                print(f"[{cnt}件] {repr(title)}")
        else:
            print("（該当なし）")

        instructions, err = generate_blog_instruction(
            saved_meta.get("keyword", args.keyword or ""), results, common_subs, title_ranks
        )
        if instructions:
            print("=== ブログ作成指示書 ===")
            print(instructions)
            blog_post, err2 = generate_blog_post(
                saved_meta.get("keyword", args.keyword or ""), instructions
            )
            if blog_post:
                print("=== 生成ブログ記事 ===")
                print(blog_post)
                path = save_blog_markdown(
                    blog_post, saved_meta.get("keyword", args.keyword or ""),
                )
                print(f"Markdownとして保存: {path}")
            else:
                logging.info("Blog generation failed: %s", err2)
        else:
            logging.info("Instruction generation failed: %s", err)

        logging.info("Finished (analysis-load mode). results=%d", len(results))
        return

    # ============ 通常モード（検索→取得→解析→保存） ============
    if not args.keyword:
        raise SystemExit("keyword が必要です（または --analysis-load を指定してください）")

    # 除外定義の読み込み
    exclude_chars, remove_trans, exclude_regex = load_excludes(args.exclude_file)
    if exclude_chars or exclude_regex:
        logging.info(
            "Excluding %d chars and %d regex patterns in analysis.",
            len(exclude_chars),
            len(exclude_regex),
        )

    logging.info('検索開始 keyword="%s" num=%d', args.keyword, args.num_results)
    urls = get_search_results(args.keyword, args.num_results, args.delay)
    session_factory = create_session
    thread_local = threading.local()
    robots_cache: Dict[str, bool] = {}
    robots_lock = threading.Lock()
    results: List[Dict] = []
    all_texts: List[str] = []
    seo_titles: List[str] = []
    skipped_total = 0

    def process_url(target_url: str):
        session = getattr(thread_local, "session", None)
        if session is None:
            session = session_factory()
            thread_local.session = session
        domain = extract_domain(target_url)
        with robots_lock:
            robots = robots_cache.get(domain)
        if robots is None:
            robots = robots_exists(session, target_url)
            with robots_lock:
                robots_cache[domain] = robots
        if not robots:
            logging.info('Skip %s robots.txt disallow', target_url)
            return None
        html = fetch_html(session, target_url)
        if not html:
            return None
        data = parse_html(html)
        word_count, top_keywords = analyze_keywords(
            data['text'], merge_threshold=args.merge_percent / 100.0
        )
        row = {
            'url': target_url,
            'domain': domain,
            'published_time': data['published_time'],
            'title': data['title'],
            'description': data['description'],
            'word_count': word_count,
            'top_keywords': top_keywords,
            'text': data['text'][:args.chars],
            'robots': robots,
            'images': data['images'],
            'links': data['links'],
            'headings': data['headings'],
        }
        logging.info('OK %s | title="%s" robots=%s', target_url, data['title'], robots)
        return row, data['text'], data['title']

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = {ex.submit(process_url, u): u for u in urls}
        for fut in as_completed(future_map):
            res = fut.result()
            if res is None:
                skipped_total += 1
                continue
            row, text, title = res
            results.append(row)
            all_texts.append(text)
            seo_titles.append(title)

    if results:
        # 正規化済み配列（分析ファイルにも保存）
        norm_texts = [
            _normalize_for_substrings(t, remove_trans, exclude_regex)[:args.analyze_chars]
            for t in all_texts
        ]
        norm_titles = []
        for t in seo_titles:
            if not t:
                norm_titles.append("")
                continue
            s = t.translate(remove_trans) if remove_trans else t
            if exclude_regex:
                for pat in exclude_regex:
                    s = pat.sub(' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            norm_titles.append(s)

        # 共通本文サブ文字列（3～12文字・最長一致優先）
        if args.analysis_mode == 'char':
            raw_subs = common_substrings_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        elif args.analysis_mode == 'hybrid':
            raw_subs = hybrid_common_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        else:
            raw_subs = common_tokens_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k * 3,
                max_doc_ratio=args.max_common_ratio,
            )
        filtered = filter_common_phrases(raw_subs, args.keyword)[:args.rank_k]
        total_docs = len(norm_texts) if norm_texts else 1
        common_subs = [
            {"text": sub, "count": cnt, "ratio": cnt / total_docs}
            for sub, cnt in filtered
        ]
        # SEOタイトルの共通部分列ランキング
        title_ranks = rank_common_titles_from_norm(
            norm_titles,
            top_k=args.rank_k,
            mode=args.analysis_mode,
        )

        logging.info("Summary: hits=%d, collected=%d, skipped=%d",
                     len(urls), len(results), skipped_total)

        unit_en = "chars" if args.analysis_mode == 'char' else "tokens"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        if common_subs:
            logging.info("Top %d COMMON SUBSTRINGS (len 3-12 %s, longest-match):", len(common_subs), unit_en)
            for item in common_subs:
                sub = item["text"]
                cnt = item["count"]
                ratio = item["ratio"]
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                logging.info("[SUB %d / %.0f%%] %r (len=%d)", cnt, ratio * 100, sub, length)
        else:
            logging.info("No common substrings found (len 3-12 %s).", unit_en)

        if title_ranks:
            logging.info("Top %d COMMON SEO TITLE SUBSTRINGS:", len(title_ranks))
            for title, cnt in title_ranks:
                logging.info("[TITLE %d] %r", cnt, title)
        else:
            logging.info("No common SEO title substrings (>=2 occurrences).")

        # ファイル書き出し
        if args.results_csv:
            write_results_csv(args.results_csv, results)

        if args.results_json:
            meta = {
                "source": "fresh_crawl",
                "keyword": args.keyword,
                "num_results_requested": args.num_results,
                "hits_total": len(urls),
                "collected": len(results),
                "skipped": skipped_total,
                "rank_k": args.rank_k,
                "analyze_chars": args.analyze_chars,
                "max_common_ratio": args.max_common_ratio,
                "analysis_mode": args.analysis_mode,
                "merge_percent": args.merge_percent,
                "exclude_chars": sorted(list(exclude_chars)),
                "exclude_chars_count": len(exclude_chars),
                "exclude_regex": [p.pattern for p in exclude_regex],
                "exclude_regex_count": len(exclude_regex),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "common_substrings": common_subs,
                "common_seo_title_substrings": title_ranks,
            }
            write_results_json(args.results_json, results, meta)

        # 分析ファイル（ローカル再集計用）を保存
        if args.analysis_save:
            meta_for_analysis = {
                "keyword": args.keyword,
                "analyze_chars": args.analyze_chars,
                "max_common_ratio": args.max_common_ratio,
                "analysis_mode": args.analysis_mode,
                "merge_percent": args.merge_percent,
                "exclude_chars": sorted(list(exclude_chars)),
                "exclude_regex": [p.pattern for p in exclude_regex],
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_analysis(args.analysis_save, meta_for_analysis, norm_texts, norm_titles, results)

        # ====== コンソール出力（従来の結果一覧：--show-pages 指定時のみ） ======
        if args.show_pages:
            for item in results:
                print("URL:", item['url'])
                print("ドメイン:", item['domain'])
                print("公開日:　", item['published_time'])
                print("SEOタイトル:", item['title'])
                print("説明:", item.get('description', ''))
                print("robots.txt:　", "あり" if item['robots'] else "なし")
                print("語数:", item['word_count'])
                print("上位キーワード:", ', '.join(f"{k['keyword']}:{k['count']}" for k in item['top_keywords']))
                print("画像数:", item.get('images'))
                print("リンク数:", item.get('links'))
                print("本文:　", item['text'])
                print("-" * 80)

        unit = "文字" if args.analysis_mode == 'char' else "トークン"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        print(f"共通本文サブ文字列（3～12{unit}, 最長一致・上位{args.rank_k}）:")
        if common_subs:
            for item in common_subs:
                sub = item["text"]
                cnt = item["count"]
                ratio = item["ratio"]
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                print(f"[{cnt}件 / {length}{unit} / {ratio:.0%}] {repr(sub)}")
        else:
            print("（該当なし）")

        print(f"共通SEOタイトルサブ文字列（3～12{unit}、上位{args.rank_k}）:")
        if title_ranks:
            for title, cnt in title_ranks:
                print(f"[{cnt}件] {repr(title)}")
        else:
            print("（該当なし）")
        instructions, err = generate_blog_instruction(
            args.keyword, results, common_subs, title_ranks
        )
        if instructions:
            print("=== ブログ作成指示書 ===")
            print(instructions)
            blog_post, err2 = generate_blog_post(args.keyword, instructions)
            if blog_post:
                print("=== 生成ブログ記事 ===")
                print(blog_post)
                path = save_blog_markdown(blog_post, args.keyword)
                print(f"Markdownとして保存: {path}")
            else:
                logging.info("Blog generation failed: %s", err2)
        else:
            logging.info("Instruction generation failed: %s", err)

        logging.info("Finished. results=%d", len(results))
    else:
        print("結果を取得できませんでした")
        logging.warning("No results (skipped=%d)", skipped_total)


if __name__ == '__main__':
    main()
