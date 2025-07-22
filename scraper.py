import argparse
import logging
from typing import List, Optional

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



def get_search_results(query: str, num_results: int) -> List[str]:
    return list(search(query, num_results=num_results))


def fetch_html(session: requests.Session, url: str, timeout: int = 10) -> Optional[str]:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning("Failed to fetch %s: %s", url, e)
        return None


def parse_html(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    text = ' '.join(p.get_text(separator=' ', strip=True) for p in soup.find_all('p'))
    time_tag = soup.find('meta', attrs={'property': 'article:published_time'})
    if time_tag and time_tag.get('content'):
        published_time = time_tag['content']
    else:
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            published_time = time_tag['datetime']
        else:
            published_time = 'N/A'
    return {
        'text': text,
        'published_time': published_time
    }


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return '.'.join(part for part in [ext.domain, ext.suffix] if part)


def main():
    parser = argparse.ArgumentParser(description='Search and scrape web pages.')
    parser.add_argument('keyword', help='Keyword to search for')
    parser.add_argument('-n', '--num-results', type=int, default=10, help='Number of results to fetch')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info('Searching for "%s"', args.keyword)
    urls = get_search_results(args.keyword, args.num_results)
    session = create_session()

    results = []
    for url in urls:
        html = fetch_html(session, url)
        if not html:
            continue
        data = parse_html(html)
        domain = extract_domain(url)
        results.append({
            'url': url,
            'domain': domain,
            'published_time': data['published_time'],
            'text': data['text'][:500]
        })

    if results:
        for item in results:
            print("URL:", item['url'])
            print("Domain:", item['domain'])
            print("Published:", item['published_time'])
            print("Text:", item['text'])
            print("-" * 80)
    else:
        print("No results fetched.")


if __name__ == '__main__':
    main()
