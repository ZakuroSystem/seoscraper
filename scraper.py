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

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from googlesearch import search
import tldextract


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
# 共通判定（トークナイザ不使用 / 3～12文字・最長一致優先）
# =========================
_PRESENT_CHAR = re.compile(r'[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]')

def _normalize_for_substrings(s: str) -> str:
    """空白を一つに圧縮し、前後の空白を除去（判定強化・ノイズ低減）。"""
    if not s:
        return ""
    # 改行やタブをスペースに、連続空白を1つに
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _iter_substrings(s: str, min_len: int, max_len: int) -> Iterable[str]:
    """長さ制約内の部分文字列をすべて生成（文字列そのまま、トークナイザ不使用）。"""
    n = len(s)
    max_len = min(max_len, n)
    for L in range(min_len, max_len + 1):
        for i in range(0, n - L + 1):
            sub = s[i:i+L]
            # 文字種フィルタ：完全な空白/記号列は除外（判定強化）
            if _PRESENT_CHAR.search(sub):
                yield sub

def common_substrings_rank(
    texts: List[str],
    min_len: int = 3,
    max_len: int = 12,
    analyze_chars: int = 5000,
    top_k: int = 15
) -> List[Tuple[str, int]]:
    """
    複数ドキュメント間の共通部分文字列を集計。
    - 3〜12文字の一致のみ対象
    - 各ドキュメント内では重複カウントしない（出現ドキュメント数ベース）
    - 同一ドキュメント集合で包含関係がある場合、最長一致を優先して短い一致を除外
    - 最終ランキングは出現ドキュメント数 desc → 長さ desc → 文字列 asc
    """
    # ドキュメントID集合での出現マップ
    sub_to_docs: Dict[str, Set[int]] = defaultdict(set)

    for doc_id, raw in enumerate(texts):
        s = _normalize_for_substrings(raw)[:analyze_chars]
        if not s:
            continue
        seen_in_doc: Set[str] = set()
        for sub in _iter_substrings(s, min_len, max_len):
            if sub in seen_in_doc:
                continue
            seen_in_doc.add(sub)
            sub_to_docs[sub].add(doc_id)

    # 出現が2ドキュメント以上のもののみ対象（"共通"）
    grouped: Dict[frozenset, List[str]] = defaultdict(list)
    for sub, docs in sub_to_docs.items():
        if len(docs) >= 2:
            grouped[frozenset(docs)].append(sub)

    # 同一ドキュメント集合ごとに最長一致優先で短い一致を除外
    filtered: List[Tuple[str, int]] = []
    for docset, subs in grouped.items():
        subs.sort(key=lambda x: (-len(x), x))  # 長い順 → 同長は辞書順
        kept: List[str] = []
        for sub in subs:
            # 既に採用済みのより長い一致に完全に含まれるならスキップ
            if any(ks.find(sub) != -1 for ks in kept):
                continue
            kept.append(sub)
        for sub in kept:
            filtered.append((sub, len(docset)))

    # ランキング整列：出現ドキュメント数 desc → 長さ desc → 文字列 asc
    filtered.sort(key=lambda t: (-t[1], -len(t[0]), t[0]))
    return filtered[:top_k]


def rank_equal_titles(titles: List[str], top_k: int = 15) -> List[Tuple[str, int]]:
    """
    SEOタイトルの完全一致ランキング。
    - 余白をトリムした完全一致で集計
    - 出現回数が2以上のものをランキング（検索で得た集合内のみ）
    """
    counter = Counter(t.strip() for t in titles if t and t.strip())
    items = [(t, c) for t, c in counter.items() if c >= 2]
    items.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return items[:top_k]


# =========================
# 結果書き出し
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


# =========================
# メイン
# =========================
def main():
    parser = argparse.ArgumentParser(description='ウェブを検索し情報を抽出するツール')
    parser.add_argument('keyword', help='検索キーワード')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='取得するURLの件数')
    parser.add_argument('--delay', type=float, default=1.0, help='リクエストの間隔(秒)')
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
    # 互換（旧オプション）: --top-k が与えられたら --rank-k に流用
    parser.add_argument('--top-k', type=int, default=None, help='[互換] ランキング件数。指定時は --rank-k を上書き')
    args = parser.parse_args()

    if args.top_k is not None:
        args.rank_k = args.top_k

    setup_logging(args.log_file, args.log_level, args.log_max_bytes, args.log_backup_count)

    logging.info('検索開始 keyword="%s" num=%d', args.keyword, args.num_results)
    urls = get_search_results(args.keyword, args.num_results, args.delay)
    session = create_session()

    results: List[Dict] = []
    robots_cache: Dict[str, bool] = {}
    all_texts: List[str] = []
    seo_titles: List[str] = []

    skipped_total = 0

    for url in urls:
        html = fetch_html(session, url)
        if not html:
            skipped_total += 1
            time.sleep(args.delay)
            continue
        data = parse_html(html)
        domain = extract_domain(url)
        if domain not in robots_cache:
            robots_cache[domain] = robots_exists(session, url)

        all_texts.append(data['text'])
        seo_titles.append(data['title'])

        row = {
            'url': url,
            'domain': domain,
            'published_time': data['published_time'],
            'title': data['title'],
            'text': data['text'][:args.chars],
            'robots': robots_cache[domain]
        }
        results.append(row)

        logging.info('OK %s | title="%s" robots=%s', url, data['title'], robots_cache[domain])
        time.sleep(args.delay)

    if results:
        # 共通本文サブ文字列（3～12文字・最長一致優先）
        common_subs = common_substrings_rank(
            all_texts, min_len=3, max_len=12,
            analyze_chars=args.analyze_chars, top_k=args.rank_k
        )
        # SEOタイトルの完全一致ランキング
        title_ranks = rank_equal_titles(seo_titles, top_k=args.rank_k)

        logging.info("Summary: hits=%d, collected=%d, skipped=%d",
                     len(urls), len(results), skipped_total)

        # ログ出力（ランキング）
        if common_subs:
            logging.info("Top %d COMMON SUBSTRINGS (len 3-12, longest-match):", len(common_subs))
            for sub, cnt in common_subs:
                logging.info("[SUB %d] %r (len=%d)", cnt, sub, len(sub))
        else:
            logging.info("No common substrings found (len 3-12).")

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
                "keyword": args.keyword,
                "num_results_requested": args.num_results,
                "hits_total": len(urls),
                "collected": len(results),
                "skipped": skipped_total,
                "rank_k": args.rank_k,
                "analyze_chars": args.analyze_chars,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "common_substrings": common_subs,
                "equal_seo_titles": title_ranks,
            }
            write_results_json(args.results_json, results, meta)

        # ====== コンソール出力（従来の結果一覧） ======
        for item in results:
            print("URL:", item['url'])
            print("ドメイン:", item['domain'])
            print("公開日:　", item['published_time'])
            print("SEOタイトル:", item['title'])
            print("robots.txt:　", "あり" if item['robots'] else "なし")
            print("本文:　", item['text'])
            print("-" * 80)

        print(f"共通本文サブ文字列（3～12文字, 最長一致・上位{args.rank_k}）:")
        if common_subs:
            for sub, cnt in common_subs:
                print(f"[{cnt}件 / {len(sub)}文字] {repr(sub)}")
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
