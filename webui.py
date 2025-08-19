from typing import Dict
from flask import Flask, render_template, request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper import (
    create_session,
    get_search_results,
    fetch_html,
    parse_html,
    extract_domain,
    robots_exists,
    common_substrings_rank,
    rank_equal_titles,
    filter_common_phrases,
    parse_exclude_lines,
)

app = Flask(__name__)


def run_analysis(keyword: str, num_results: int, delay: float, rank_k: int,
                 analyze_chars: int, max_common_ratio: float, analysis_mode: str,
                 workers: int, remove_trans, remove_patterns):
    """検索と解析を実行し結果を返す"""
    urls = get_search_results(keyword, num_results, delay)
    robots_cache: Dict[str, bool] = {}
    robots_lock = threading.Lock()
    results = []
    texts = []
    titles = []

    def process_url(url: str):
        session = create_session()
        domain = extract_domain(url)
        with robots_lock:
            robots = robots_cache.get(domain)
        if robots is None:
            robots = robots_exists(session, url)
            with robots_lock:
                robots_cache[domain] = robots
        if not robots:
            return None
        html = fetch_html(session, url)
        if not html:
            return None
        data = parse_html(html)
        row = {
            "url": url,
            "domain": domain,
            "published_time": data["published_time"],
            "title": data["title"],
            "robots": robots,
            "text": data["text"][:analyze_chars],
        }
        return row, data["text"], data["title"]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_url, u) for u in urls]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                row, text, title = res
                results.append(row)
                texts.append(text)
                titles.append(title)

    common_subs = common_substrings_rank(
        texts,
        analyze_chars=analyze_chars,
        top_k=rank_k,
        remove_trans=remove_trans,
        remove_patterns=remove_patterns,
        max_doc_ratio=max_common_ratio,
        mode=analysis_mode,
    )
    common_subs = filter_common_phrases(common_subs, keyword)
    title_ranks = rank_equal_titles(titles, top_k=rank_k,
                                   remove_trans=remove_trans,
                                   remove_patterns=remove_patterns)
    return results, common_subs, title_ranks


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keyword = request.form.get('keyword', '')
        if keyword:
            num_results = int(request.form.get('num_results', 10))
            delay = float(request.form.get('delay', 0.1))
            workers = int(request.form.get('workers', 10))
            analyze_chars = int(request.form.get('analyze_chars', 5000))
            rank_k = int(request.form.get('rank_k', 15))
            max_common_ratio = float(request.form.get('max_common_ratio', 0.8))
            analysis_mode = request.form.get('analysis_mode', 'tiktoken')
            excl_lines = request.form.get('exclude_patterns', '').splitlines()
            _, remove_trans, remove_patterns = parse_exclude_lines(excl_lines)
            results, common_subs, title_ranks = run_analysis(
                keyword,
                num_results,
                delay,
                rank_k,
                analyze_chars,
                max_common_ratio,
                analysis_mode,
                workers,
                remove_trans,
                remove_patterns,
            )
            return render_template(
                'index.html',
                results=results,
                common_subs=common_subs,
                title_ranks=title_ranks,
                form=request.form,
            )
    return render_template('index.html', results=None, form=None)


if __name__ == '__main__':
    app.run(port=5000)

