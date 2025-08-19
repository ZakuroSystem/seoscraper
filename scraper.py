import argparse
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import List, Optional, Tuple, Dict, Set, Iterable
from collections import Counter, defaultdict
import re
import json
import csv
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

    logging.debug("Parsed title=%s published=%s", seo_title, published_time)
    return {
        'text': text,
        'published_time': published_time,
        'title': seo_title
    }


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return '.'.join(part for part in [ext.domain, ext.suffix] if part)


# =========================
# 除外文字（txt）の読み込み
# =========================
def load_exclude_chars(path: Optional[str]) -> Tuple[Set[str], Optional[Dict[int, None]]]:
    """
    UTF-8 (BOM可) のテキストファイルから除外文字集合を作成。
    - 改行(\n, \r)は無視
    - ファイル中に現れる各文字を 1 文字単位で除外対象にする
    - 返り値は（除外集合, str.translate 用の削除テーブル）
    """
    if not path:
        return set(), None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = f.read()
        chars = {ch for ch in data if ch not in ("\n", "\r")}
        if not chars:
            logging.info("Exclude-chars file is empty: %s", path)
            return set(), None
        delete_table = str.maketrans("", "", "".join(sorted(chars)))
        logging.info("Loaded exclude chars: %d from %s", len(chars), path)
        return chars, delete_table
    except Exception as e:
        logging.error("Failed to load exclude chars file %s: %s", path, e)
        return set(), None


# =========================
# 共通判定（文字列/トークン単位, 3～12長・最長一致優先）
# =========================
_PRESENT_CHAR = re.compile(r'[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]')

def _normalize_for_substrings(s: str, remove_trans: Optional[Dict[int, None]] = None) -> str:
    """除外文字の削除 → 連続空白を1つに圧縮 → 前後の空白を除去。"""
    if not s:
        return ""
    if remove_trans:
        s = s.translate(remove_trans)
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
    return [(enc.decode(list(seq)), cnt) for seq, cnt in filtered[:top_k]]


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

def rank_equal_titles_from_norm(norm_titles: List[str], top_k: int = 15) -> List[Tuple[str, int]]:
    """
    すでに正規化済み（除外適用＋空白正規化済み）のSEOタイトル配列から完全一致ランキング。
    """
    counter = Counter(t for t in norm_titles if t)
    items = [(t, c) for t, c in counter.items() if c >= 2]
    items.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return items[:top_k]

def common_substrings_rank(
    texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    analyze_chars: int = 5000,
    top_k: int = 15,
    remove_trans: Optional[Dict[int, None]] = None,
    max_doc_ratio: float = 0.8,
    mode: str = "tiktoken",
    encoding_name: str = "cl100k_base",
) -> List[Tuple[str, int]]:
    """（オンライン計算用）正規化→切り詰め→共通部分列抽出。
    mode に応じて文字列 or トークン列で判定。"""
    norm_texts = [_normalize_for_substrings(t, remove_trans)[:analyze_chars] for t in texts]
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

def rank_equal_titles(
    titles: List[str],
    top_k: int = 15,
    remove_trans: Optional[Dict[int, None]] = None
) -> List[Tuple[str, int]]:
    """（オンライン計算用）正規化→完全一致ランキング。"""
    normed = []
    for t in titles:
        if not t:
            continue
        s = t
        if remove_trans:
            s = s.translate(remove_trans)
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            normed.append(s)
    return rank_equal_titles_from_norm(normed, top_k=top_k)


# =========================
# 結果・分析ファイルの書き出し／読み込み
# =========================
def write_results_csv(path: str, rows: List[Dict]):
    fieldnames = ["url", "domain", "published_time", "title", "robots", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
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
    parser.add_argument('--delay', type=float, default=0.5, help='各リクエスト前の待機秒数(デフォルト0.5)')
    parser.add_argument('--workers', type=int, default=5, help='同時リクエスト数')
    parser.add_argument('--chars', type=int, default=1000, help='本文の表示文字数')
    # ログ
    parser.add_argument('--log-file', default=None, help='ログ出力先ファイル（指定しない場合はコンソールのみ）')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='ログレベル')
    parser.add_argument('--log-max-bytes', type=int, default=5*1024*1024, help='ローテーション閾値（バイト）')
    parser.add_argument('--log-backup-count', type=int, default=3, help='ローテーション世代数')
    # 出力
    parser.add_argument('--results-csv', default=None, help='結果CSVのパス')
    parser.add_argument('--results-json', default=None, help='結果JSONのパス')
    # 共通判定パラメータ
    parser.add_argument('--rank-k', type=int, default=15, help='ランキングの表示件数（共通本文サブ文字列 / 一致SEOタイトル）')
    parser.add_argument('--analyze-chars', type=int, default=5000, help='共通判定に用いる本文の先頭文字数（デフォルト5000）')
    parser.add_argument('--exclude-chars-file', default=None, help='共通判定前に除去する文字の一覧テキストファイル（UTF-8/BOM可）。各文字をそのまま列挙（改行は無視）。')
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
        saved_analyze_chars = saved_meta.get("analyze_chars")
        saved_mode = saved_meta.get("analysis_mode")
        # 現在の除外ファイルを読み込んだ場合は一致比較
        exclude_chars, remove_trans = load_exclude_chars(args.exclude_chars_file)
        if exclude_chars and exclude_chars != saved_excl:
            logging.warning("Exclude chars differ from analysis file. Recalc uses SAVED normalization.")
        if args.analyze_chars and saved_analyze_chars and args.analyze_chars != saved_analyze_chars:
            logging.warning("analyze_chars differs from analysis file. Recalc uses SAVED truncation.")
        if saved_mode and args.analysis_mode != saved_mode:
            logging.warning("analysis_mode differs from analysis file. Recalc uses %s", args.analysis_mode)

        norm_texts = data.get("norm_texts", [])
        norm_titles = data.get("norm_titles", [])
        results = data.get("results", [])

        # ローカル再集計（top-k 変更だけなら超高速）
        if args.analysis_mode == 'char':
            common_subs = common_substrings_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        elif args.analysis_mode == 'hybrid':
            common_subs = hybrid_common_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        else:
            common_subs = common_tokens_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        common_subs = filter_common_phrases(common_subs, saved_meta.get("keyword", ""))
        title_ranks = rank_equal_titles_from_norm(norm_titles, top_k=args.rank_k)

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
                "exclude_chars_count": len(saved_excl),
                "common_substrings": common_subs,
                "equal_seo_titles": title_ranks,
            }
            write_results_json(args.results_json, results, meta)

        # 従来の出力
        for item in results:
            print("URL:", item['url'])
            print("ドメイン:", item['domain'])
            print("公開日:　", item['published_time'])
            print("SEOタイトル:", item['title'])
            print("robots.txt:　", "あり" if item['robots'] else "なし")
            print("本文:　", item['text'])
            print("-" * 80)

        unit = "文字" if args.analysis_mode == 'char' else "トークン"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        print(f"共通本文サブ文字列（3～12{unit}, 最長一致・上位{args.rank_k}）:")
        if common_subs:
            for sub, cnt in common_subs:
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                print(f"[{cnt}件 / {length}{unit}] {repr(sub)}")
        else:
            print("（該当なし）")

        print(f"一致SEOタイトル（完全一致・上位{args.rank_k}）:")
        if title_ranks:
            for title, cnt in title_ranks:
                print(f"[{cnt}件] {repr(title)}")
        else:
            print("（該当なし）")

        logging.info("Finished (analysis-load mode). results=%d", len(results))
        return

    # ============ 通常モード（検索→取得→解析→保存） ============
    if not args.keyword:
        raise SystemExit("keyword が必要です（または --analysis-load を指定してください）")

    # 除外文字の読み込み
    exclude_chars, remove_trans = load_exclude_chars(args.exclude_chars_file)
    if exclude_chars:
        logging.info("Excluding %d characters in analysis.", len(exclude_chars))

    logging.info('検索開始 keyword="%s" num=%d', args.keyword, args.num_results)
    urls = get_search_results(args.keyword, args.num_results, args.delay)
    session_factory = create_session
    robots_cache: Dict[str, bool] = {}
    robots_lock = threading.Lock()
    results: List[Dict] = []
    all_texts: List[str] = []
    seo_titles: List[str] = []
    skipped_total = 0

    def process_url(target_url: str):
        time.sleep(args.delay)
        session = session_factory()
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
        row = {
            'url': target_url,
            'domain': domain,
            'published_time': data['published_time'],
            'title': data['title'],
            'text': data['text'][:args.chars],
            'robots': robots,
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
        norm_texts = [_normalize_for_substrings(t, remove_trans)[:args.analyze_chars] for t in all_texts]
        norm_titles = []
        for t in seo_titles:
            if not t:
                norm_titles.append("")
                continue
            s = t.translate(remove_trans) if remove_trans else t
            s = re.sub(r'\s+', ' ', s).strip()
            norm_titles.append(s)

        # 共通本文サブ文字列（3～12文字・最長一致優先）
        if args.analysis_mode == 'char':
            common_subs = common_substrings_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        elif args.analysis_mode == 'hybrid':
            common_subs = hybrid_common_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        else:
            common_subs = common_tokens_rank_from_norm(
                norm_texts,
                min_len=3,
                max_len=12,
                top_k=args.rank_k,
                max_doc_ratio=args.max_common_ratio,
            )
        common_subs = filter_common_phrases(common_subs, args.keyword)
        # SEOタイトルの完全一致ランキング
        title_ranks = rank_equal_titles_from_norm(norm_titles, top_k=args.rank_k)

        logging.info("Summary: hits=%d, collected=%d, skipped=%d",
                     len(urls), len(results), skipped_total)

        unit_en = "chars" if args.analysis_mode == 'char' else "tokens"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        if common_subs:
            logging.info("Top %d COMMON SUBSTRINGS (len 3-12 %s, longest-match):", len(common_subs), unit_en)
            for sub, cnt in common_subs:
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                logging.info("[SUB %d] %r (len=%d)", cnt, sub, length)
        else:
            logging.info("No common substrings found (len 3-12 %s).", unit_en)

        if title_ranks:
            logging.info("Top %d EQUAL SEO TITLES:", len(title_ranks))
            for title, cnt in title_ranks:
                logging.info("[TITLE %d] %r", cnt, title)
        else:
            logging.info("No equal SEO titles (>=2 occurrences).")

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
                "exclude_chars": sorted(list(exclude_chars)),
                "exclude_chars_count": len(exclude_chars),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "common_substrings": common_subs,
                "equal_seo_titles": title_ranks,
            }
            write_results_json(args.results_json, results, meta)

        # 分析ファイル（ローカル再集計用）を保存
        if args.analysis_save:
            meta_for_analysis = {
                "keyword": args.keyword,
                "analyze_chars": args.analyze_chars,
                "max_common_ratio": args.max_common_ratio,
                "analysis_mode": args.analysis_mode,
                "exclude_chars": sorted(list(exclude_chars)),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_analysis(args.analysis_save, meta_for_analysis, norm_texts, norm_titles, results)

        # ====== コンソール出力（従来の結果一覧） ======
        for item in results:
            print("URL:", item['url'])
            print("ドメイン:", item['domain'])
            print("公開日:　", item['published_time'])
            print("SEOタイトル:", item['title'])
            print("robots.txt:　", "あり" if item['robots'] else "なし")
            print("本文:　", item['text'])
            print("-" * 80)

        unit = "文字" if args.analysis_mode == 'char' else "トークン"
        enc = tiktoken.get_encoding("cl100k_base") if args.analysis_mode != 'char' else None
        print(f"共通本文サブ文字列（3～12{unit}, 最長一致・上位{args.rank_k}）:")
        if common_subs:
            for sub, cnt in common_subs:
                length = len(sub) if args.analysis_mode == 'char' else len(enc.encode(sub))
                print(f"[{cnt}件 / {length}{unit}] {repr(sub)}")
        else:
            print("（該当なし）")

        print(f"一致SEOタイトル（完全一致・上位{args.rank_k}）:")
        if title_ranks:
            for title, cnt in title_ranks:
                print(f"[{cnt}件] {repr(title)}")
        else:
            print("（該当なし）")

        logging.info("Finished. results=%d", len(results))
    else:
        print("結果を取得できませんでした")
        logging.warning("No results (skipped=%d)", skipped_total)


if __name__ == '__main__':
    main()
