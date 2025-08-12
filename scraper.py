import argparse
import logging
import time
from typing import List, Optional
from collections import Counter
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from googlesearch import search
import tldextract


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
        return list(search(query, num_results=num_results, sleep_interval=pause))
    except Exception as e:
        logging.error("Search failed: %s", e)
        return []


# ---------- ここから: 文字化け対策 ----------

_MOJIBAKE_PATTERNS = [
    r"Ã.", r"Â.", r"â..", r"ðŸ", r"�",   # ラテン系/絵文字崩れ/置換文字
    r"ã‚", r"ãƒ", r"ã„", r"ãŒ",          # UTF-8→SJIS/EUC 誤解読で出やすい
    r"å.", r"æ.", r"œ"                   # よく見る æ/œ/å 系
]
_MOJIBAKE_REGEX = re.compile("|".join(_MOJIBAKE_PATTERNS))

def mojibake_score(text: str) -> float:
    """ざっくり文字化けスコア（テキスト長で正規化）。0に近いほど正常。"""
    if not text:
        return 1.0
    hits = len(_MOJIBAKE_REGEX.findall(text))
    # 置換文字（�）は重めにカウント
    hits += text.count("�") * 2
    # 連続的に現れる場合を少し加点
    hits += len(re.findall(r"(Ã|Â|â){3,}", text)) * 3
    return hits / max(len(text), 1)

def _find_meta_charset(head_bytes: bytes) -> Optional[str]:
    """<meta charset=...> や http-equiv の宣言を先頭2KBから拾う"""
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

def best_decode(response: requests.Response) -> tuple[str, str, float]:
    """複数エンコーディングで試し、最も文字化けスコアが低いテキストを返す。"""
    raw = response.content
    # 先頭だけ抽出（メタ検索用）
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
            # 一部だけ読めるケースもあるので replace で試す
            txt = raw.decode(enc, errors="replace")
        score = mojibake_score(txt[:20000])  # 先頭2万文字で十分
        if score < best_score:
            best_txt, best_enc, best_score = txt, enc, score
        # スコアが十分小さければ早期終了
        if best_score < 0.0015:
            break
    return best_txt, best_enc, best_score

def fetch_html(session: requests.Session, url: str, timeout: int = 10) -> Optional[str]:
    """複数エンコーディングで再デコードし、文字化けなら None を返す。"""
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        text, used_enc, score = best_decode(resp)
        # しきい値: 経験的に 0.008 以上は読みにくいことが多い
        if score >= 0.008:
            logging.info("Skip (mojibake likely, score=%.4f, enc=%s): %s", score, used_enc, url)
            return None
        # requests.text は不要（自前で最良を採用）
        return text
    except Exception as e:
        logging.warning("Failed to fetch %s: %s", url, e)
        return None

# ---------- ここまで: 文字化け対策 ----------


def robots_exists(session: requests.Session, url: str) -> bool:
    """Return True if robots.txt exists for the URL's domain."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = session.get(robots_url, timeout=5)
        return resp.status_code == 200
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
    return {
        'text': text,
        'published_time': published_time,
        'title': seo_title
    }


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return '.'.join(part for part in [ext.domain, ext.suffix] if part)


def analyze_common_phrases(texts: List[str], top_n: int = 10) -> List[tuple]:
    """Return top common words across all texts."""
    tokens: List[str] = []
    for text in texts:
        tokens.extend(re.findall(r'\b\w+\b', text.lower()))
    counter = Counter(tokens)
    return counter.most_common(top_n)


def main():
    parser = argparse.ArgumentParser(description='ウェブを検索し情報を抽出するツール')
    parser.add_argument('keyword', help='検索キーワード')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='取得するURLの件数')
    parser.add_argument('--delay', type=float, default=1.0, help='リクエストの間隔(秒)')
    parser.add_argument('--chars', type=int, default=1000, help='表示する文字数')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info('「%s」を検索中', args.keyword)
    urls = get_search_results(args.keyword, args.num_results, args.delay)
    session = create_session()

    results = []
    robots_cache = {}
    all_texts: List[str] = []
    for url in urls:
        html = fetch_html(session, url)
        if not html:
            time.sleep(args.delay)
            continue
        data = parse_html(html)
        domain = extract_domain(url)
        if domain not in robots_cache:
            robots_cache[domain] = robots_exists(session, url)
        all_texts.append(data['text'])
        results.append({
            'url': url,
            'domain': domain,
            'published_time': data['published_time'],
            'title': data['title'],
            'text': data['text'][:args.chars],
            'robots': robots_cache[domain]
        })
        time.sleep(args.delay)

    if results:
        for item in results:
            print("URL:", item['url'])
            print("ドメイン:", item['domain'])
            print("公開日:　", item['published_time'])
            print("SEOタイトル:", item['title'])
            print("robots.txt:　", "あり" if item['robots'] else "なし")
            print("本文:　", item['text'])
            print("-" * 80)
        counts = analyze_common_phrases(all_texts)
        print("共通ワード出現回数:")
        for word, cnt in counts:
            print(f"{word}: {cnt}")
    else:
        print("結果を取得できませんでした")


if __name__ == '__main__':
    main()
