import argparse
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import List, Optional, Tuple, Dict
from collections import Counter
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


def setup_logging(log_file: Optional[str], level: str, max_bytes: int, backup_count: int):
    """コンソールとファイルにログを出力（ローテーション対応）"""
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


def create_session() -> requests.Session:
    """リトライ付きHTTPセッション作成"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def get_search_results(query: str, num_results: int, pause: float) -> List[str]:
    """Google検索でURL一覧を取得"""
    try:
        urls = list(search(query, num_results=num_results, sleep_interval=pause))
        logging.info("Search ok: %s (hits=%d)", query, len(urls))
        return urls
    except Exception as e:
        logging.error("Search failed: %s", e)
        return []


# ---------- 文字化け対策 ----------
_MOJIBAKE_PATTERNS = [
    r"Ã.", r"Â.", r"â..", r"ðŸ", r"�",
    r"ã‚", r"ãƒ", r"ã„", r"ãŒ",
    r"å.", r"æ.", r"œ"
]
_MOJIBAKE_REGEX = re.compile("|".join(_MOJIBAKE_PATTERNS))

def mojibake_score(text: str) -> float:
    if not text:
        return 1.0
    hits = len(_MOJIBAKE_REGEX.findall(text))
    hits += text.count("�") * 2
    hits += len(re.findall(r"(Ã|Â|â){3,}", text)) * 3
    return hits / max(len(text), 1)

def _find_meta_charset(head_bytes: bytes) -> Optional[str]:
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

def _unique_clean(seq):
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
    raw = response.content
    head = raw[:2048]
    candidates = _unique_clean([
        "utf-8",
        response.encoding,
        getattr(response, "apparent_encoding", None),
        _find_meta_charset(head),
        "cp932", "shift_jis", "euc-jp", "iso-2022-jp",
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
# ---------- 文字化け対策 ここまで ----------


def robots_exists(session: requests.Session, url: str) -> bool:
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


def analyze_common_exact(strings: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """文字列全体の完全一致でカウント"""
    counter = Counter(s.strip() for s in strings if s and s.strip())
    return counter.most_common(top_n)


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


def main():
    parser = argparse.ArgumentParser(description='ウェブを検索し情報を抽出するツール')
    parser.add_argument('keyword', help='検索キーワード')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='取得するURLの件数')
    parser.add_argument('--delay', type=float, default=1.0, help='リクエストの間隔(秒)')
    parser.add_argument('--chars', type=int, default=1000, help='表示する文字数')
    parser.add_argument('--log-file', default=None, help='ログ出力先ファイル')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='ログレベル')
    parser.add_argument('--log-max-bytes', type=int, default=5*1024*1024, help='ログローテーション閾値')
    parser.add_argument('--log-backup-count', type=int, default=3, help='ログローテーション世代数')
    parser.add_argument('--results-csv', default=None, help='結果CSVのパス')
    parser.add_argument('--results-json', default=None, help='結果JSONのパス')
    parser.add_argument('--top-k', type=int, default=15, help='共通項目の上位件数')
    args = parser.parse_args()

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
        common_body = analyze_common_exact(all_texts, top_n=args.top_k)
        common_titles = analyze_common_exact(seo_titles, top_n=args.top_k)

        logging.info("Summary: hits=%d, collected=%d, skipped=%d",
                     len(urls), len(results), skipped_total)

        logging.info("Top %d COMMON BODY TEXTS:", args.top_k)
        for text, cnt in common_body:
            logging.info("[BODY %d] %r", cnt, text)

        logging.info("Top %d COMMON SEO TITLES:", args.top_k)
        for title, cnt in common_titles:
            logging.info("[TITLE %d] %r", cnt, title)

        robots_true = sum(1 for r in results if r["robots"])
        robots_false = len(results) - robots_true
        logging.info("robots.txt present: %d / absent: %d", robots_true, robots_false)

        if args.results_csv:
            write_results_csv(args.results_csv, results)
        if args.results_json:
            meta = {
                "keyword": args.keyword,
                "num_results_requested": args.num_results,
                "hits_total": len(urls),
                "collected": len(results),
                "skipped": skipped_total,
                "top_body_exact": common_body,
                "top_title_exact": common_titles,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            write_results_json(args.results_json, results, meta)

        for item in results:
            print("URL:", item['url'])
            print("ドメイン:", item['domain'])
            print("公開日:　", item['published_time'])
            print("SEOタイトル:", item['title'])
            print("robots.txt:　", "あり" if item['robots'] else "なし")
            print("本文:　", item['text'])
            print("-" * 80)

        print("共通本文（完全一致）:")
        for text, cnt in common_body:
            print(f"[{cnt}件] {repr(text)}")

        print("共通SEOタイトル（完全一致）:")
        for title, cnt in common_titles:
            print(f"[{cnt}件] {repr(title)}")

        logging.info("Finished. results=%d", len(results))
    else:
        print("結果を取得できませんでした")
        logging.warning("No results (skipped=%d)", skipped_total)


if __name__ == '__main__':
    main()
