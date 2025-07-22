import argparse
import logging
import time
from typing import List, Optional
from collections import Counter
import re
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


def fetch_html(session: requests.Session, url: str, timeout: int = 10) -> Optional[str]:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning("Failed to fetch %s: %s", url, e)
        return None


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
    parser = argparse.ArgumentParser(description='\u30a6\u30a7\u30d6\u3092\u691c\u7d22\u3057\u60c5\u5831\u3092\u62bd\u51fa\u3059\u308b\u30c4\u30fc\u30eb')
    parser.add_argument('keyword', help='\u691c\u7d22\u30ad\u30fc\u30ef\u30fc\u30c9')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='\u53d6\u5f97\u3059\u308bURL\u306e\u4ef6\u6570')
    parser.add_argument('--delay', type=float, default=1.0, help='\u30ea\u30af\u30a8\u30b9\u30c8\u306e\u9593\u9694(\u79d2)')
    parser.add_argument('--chars', type=int, default=1000, help='\u8868\u793a\u3059\u308b\u6587\u5b57\u6570')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info('\u300c%s\u300d\u3092\u691c\u7d22\u4e2d', args.keyword)
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
            print("\u30c9\u30e1\u30a4\u30f3:", item['domain'])
            print("\u516c\u958b\u65e5:\u3000", item['published_time'])
            print("SEO\u30bf\u30a4\u30c8\u30eb:", item['title'])
            print("robots.txt:\u3000", "\u3042\u308a" if item['robots'] else "\u306a\u3057")
            print("\u6587\u672c:\u3000", item['text'])
            print("-" * 80)
        counts = analyze_common_phrases(all_texts)
        print("\u5171\u901a\u30ef\u30fc\u30c9\u51fa\u73fe\u56de\u6570:")
        for word, cnt in counts:
            print(f"{word}: {cnt}")
    else:
        print("\u7d50\u679c\u3092\u53d6\u5f97\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f")


if __name__ == '__main__':
    main()
