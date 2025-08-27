from typing import Dict
from flask import Flask, render_template, request, url_for
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import markdown

from scraper import (
    create_session,
    get_search_results,
    fetch_html,
    parse_html,
    extract_domain,
    robots_exists,
    common_substrings_rank,
    rank_common_titles,
    filter_common_phrases,
    parse_exclude_lines,
    analyze_keywords,
    generate_blog_instruction,
    generate_blog_post,
    save_blog_markdown,
    save_report_markdown,
    save_scrape_json,
    have_ollama_model,
)

app = Flask(__name__)

last_state = {}


def get_histories():
    base = os.path.join(os.path.dirname(__file__), 'static')
    def list_dir(sub, ext):
        dir_path = os.path.join(base, sub)
        if not os.path.isdir(dir_path):
            return []
        files = [f for f in os.listdir(dir_path) if f.endswith(ext)]
        return sorted(files, reverse=True)
    return {
        'scrapes': list_dir('scrapes', '.json'),
        'reports': list_dir('reports', '.md'),
        'blogs': list_dir('blogs', '.md'),
    }


def run_analysis(
    keyword: str,
    num_results: int,
    delay: float,
    rank_k: int,
    analyze_chars: int,
    max_common_ratio: float,
    analysis_mode: str,
    workers: int,
    remove_trans,
    remove_patterns,
    merge_percent: float,
    generate_report: bool,
    generate_blog: bool,
    report_prompt: str = "",
    blog_prompt: str = "",
):
    """検索と解析を実行し結果を返す"""
    logs = []
    logs_lock = threading.Lock()

    with logs_lock:
        logs.append(f"Google検索に問い合わせ中: {keyword}")
    urls = get_search_results(keyword, num_results, delay)
    with logs_lock:
        logs.append(f"検索結果を{len(urls)}件取得しました")
    robots_cache: Dict[str, bool] = {}
    robots_lock = threading.Lock()
    results = []
    texts = []
    titles = []
    thread_local = threading.local()

    def process_url(url: str):
        session = getattr(thread_local, "session", None)
        if session is None:
            session = create_session()
            thread_local.session = session
        with logs_lock:
            logs.append(f"スクレイピング開始: {url}")
        domain = extract_domain(url)
        with robots_lock:
            robots = robots_cache.get(domain)
        if robots is None:
            robots = robots_exists(session, url)
            with robots_lock:
                robots_cache[domain] = robots
        if not robots:
            with logs_lock:
                logs.append(f"robots.txtでアクセス拒否: {url}")
            return None
        html = fetch_html(session, url)
        if not html:
            with logs_lock:
                logs.append(f"取得失敗: {url}")
            return None
        data = parse_html(html)
        word_count, top_keywords = analyze_keywords(
            data["text"], merge_threshold=merge_percent / 100.0
        )
        row = {
            "url": url,
            "domain": domain,
            "published_time": data["published_time"],
            "title": data["title"],
            "description": data["description"],
            "robots": robots,
            "images": data["images"],
            "links": data["links"],
            "word_count": word_count,
            "top_keywords": top_keywords,
            "text": data["text"][:analyze_chars],
        }
        with logs_lock:
            logs.append(f"スクレイピング完了: {url}")
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

    # まず多めに候補を取得し、フィルタ後に上位 rank_k 件へ絞り込む
    raw_subs = common_substrings_rank(
        texts,
        analyze_chars=analyze_chars,
        top_k=rank_k * 3,
        remove_trans=remove_trans,
        remove_patterns=remove_patterns,
        max_doc_ratio=max_common_ratio,
        mode=analysis_mode,
    )
    filtered = filter_common_phrases(raw_subs, keyword)[:rank_k]
    total_docs = len(texts) if texts else 1
    common_subs = [
        {"text": sub, "count": cnt, "ratio": cnt / total_docs}
        for sub, cnt in filtered
    ]
    title_ranks = rank_common_titles(
        titles,
        top_k=rank_k,
        remove_trans=remove_trans,
        remove_patterns=remove_patterns,
        mode=analysis_mode,
    )
    instructions = None
    if generate_report:
        if have_ollama_model("gpt-oss:20b"):
            with logs_lock:
                logs.append("Ollamaで指示書生成をリクエストしています")
            instructions, err = generate_blog_instruction(
                keyword, results, common_subs, title_ranks, report_prompt
            )
            if instructions:
                with logs_lock:
                    logs.append("指示書を生成しました")
            else:
                with logs_lock:
                    logs.append(f"指示書生成失敗: {err}")
        else:
            with logs_lock:
                logs.append("gpt-oss:20bが見つからないため指示書生成をスキップしました")
    else:
        with logs_lock:
            logs.append("指示書生成をスキップしました")
    blog_post = None
    blog_file = None
    if generate_blog:
        if instructions:
            if have_ollama_model("gpt-oss:20b"):
                try:
                    with logs_lock:
                        logs.append("Ollamaでブログ生成をリクエストしています")
                    blog_post, err = generate_blog_post(keyword, instructions, blog_prompt)
                    if blog_post:
                        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'blogs')
                        path = save_blog_markdown(blog_post, keyword, directory=static_dir)
                        blog_file = os.path.basename(path)
                        with logs_lock:
                            logs.append("ブログ記事を保存しました")
                    else:
                        with logs_lock:
                            logs.append(f"ブログ生成失敗: {err}")
                except Exception as e:
                    with logs_lock:
                        logs.append(f"ブログ生成エラー: {e}")
            else:
                with logs_lock:
                    logs.append("gpt-oss:20bが見つからないためブログ生成をスキップしました")
        else:
            with logs_lock:
                logs.append("指示書がないためブログ生成をスキップしました")
    else:
        with logs_lock:
            logs.append("ブログ生成をスキップしました")
    return results, common_subs, title_ranks, instructions, blog_post, blog_file, logs


@app.route('/', methods=['GET', 'POST'])
def index():
    global last_state
    if request.method == 'POST':
        action = request.form.get('action', 'scrape')
        if action == 'scrape':
            keyword = request.form.get('keyword', '')
            if keyword:
                num_results = int(request.form.get('num_results', 10))
                delay = float(request.form.get('delay', 0.1))
                workers = int(request.form.get('workers', 10))
                analyze_chars = int(request.form.get('analyze_chars', 5000))
                rank_k = int(request.form.get('rank_k', 15))
                max_common_ratio = float(request.form.get('max_common_ratio', 0.8))
                analysis_mode = request.form.get('analysis_mode', 'tiktoken')
                merge_percent = float(request.form.get('merge_percent', 0.0))
                excl_lines = request.form.get('exclude_patterns', '').splitlines()
                _, remove_trans, remove_patterns = parse_exclude_lines(excl_lines)
                results, common_subs, title_ranks, _, _, _, logs = run_analysis(
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
                    merge_percent,
                    generate_report=False,
                    generate_blog=False,
                )
                static_dir = os.path.join(os.path.dirname(__file__), 'static', 'scrapes')
                save_scrape_json(results, keyword, directory=static_dir)
                last_state = {
                    'keyword': keyword,
                    'results': results,
                    'common_subs': common_subs,
                    'title_ranks': title_ranks,
                    'logs': logs,
                    'form': request.form,
                    'instructions': None,
                    'instructions_html': None,
                    'report_file': None,
                    'blog_post': None,
                    'blog_html': None,
                    'blog_file': None,
                    'report_prompt': '',
                    'blog_prompt': '',
                }
                hist = get_histories()
                return render_template(
                    'index.html',
                    results=results,
                    common_subs=common_subs,
                    title_ranks=title_ranks,
                    instructions=None,
                    instructions_html=None,
                    report_file=None,
                    blog_post=None,
                    blog_html=None,
                    blog_file=None,
                    logs=logs,
                    form=request.form,
                    report_prompt='',
                    blog_prompt='',
                    scrape_history=hist['scrapes'],
                    report_history=hist['reports'],
                    blog_history=hist['blogs'],
                )
        elif action == 'report' and last_state.get('results'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            report_prompt = request.form.get('report_prompt', '')
            if have_ollama_model("gpt-oss:20b"):
                logs.append("Ollamaで指示書生成をリクエストしています")
                instructions, err = generate_blog_instruction(
                    keyword,
                    last_state['results'],
                    last_state['common_subs'],
                    last_state['title_ranks'],
                    report_prompt,
                )
                if instructions:
                    static_dir = os.path.join(os.path.dirname(__file__), 'static', 'reports')
                    path = save_report_markdown(instructions, keyword, directory=static_dir)
                    report_file = os.path.basename(path)
                    logs.append("レポートを保存しました")
                else:
                    report_file = None
                    logs.append(f"指示書生成失敗: {err}")
            else:
                instructions = None
                report_file = None
                logs.append("gpt-oss:20bが見つからないため指示書生成をスキップしました")
            instructions_html = markdown.markdown(instructions) if instructions else None
            last_state.update({'instructions': instructions, 'instructions_html': instructions_html, 'report_file': report_file, 'logs': logs, 'report_prompt': report_prompt})
            hist = get_histories()
            return render_template(
                'index.html',
                results=last_state['results'],
                common_subs=last_state['common_subs'],
                title_ranks=last_state['title_ranks'],
                instructions=instructions,
                instructions_html=instructions_html,
                report_file=report_file,
                blog_post=None,
                blog_html=None,
                blog_file=None,
                logs=logs,
                form=last_state.get('form'),
                report_prompt=report_prompt,
                blog_prompt='',
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
        elif action == 'blog' and last_state.get('instructions'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            blog_prompt = request.form.get('blog_prompt', '')
            if have_ollama_model("gpt-oss:20b"):
                logs.append("Ollamaでブログ生成をリクエストしています")
                blog_post, err = generate_blog_post(keyword, last_state['instructions'], blog_prompt)
                if blog_post:
                    static_dir = os.path.join(os.path.dirname(__file__), 'static', 'blogs')
                    path = save_blog_markdown(blog_post, keyword, directory=static_dir)
                    blog_file = os.path.basename(path)
                    logs.append("ブログ記事を保存しました")
                else:
                    blog_file = None
                    logs.append(f"ブログ生成失敗: {err}")
            else:
                blog_post = None
                blog_file = None
                logs.append("gpt-oss:20bが見つからないためブログ生成をスキップしました")
            blog_html = markdown.markdown(blog_post) if blog_post else None
            last_state.update({'blog_post': blog_post, 'blog_html': blog_html, 'blog_file': blog_file, 'logs': logs, 'blog_prompt': blog_prompt})
            hist = get_histories()
            return render_template(
                'index.html',
                results=last_state['results'],
                common_subs=last_state['common_subs'],
                title_ranks=last_state['title_ranks'],
                instructions=last_state.get('instructions'),
                instructions_html=last_state.get('instructions_html'),
                report_file=last_state.get('report_file'),
                blog_post=blog_post,
                blog_html=blog_html,
                blog_file=blog_file,
                logs=logs,
                form=last_state.get('form'),
                report_prompt=last_state.get('report_prompt', ''),
                blog_prompt=blog_prompt,
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
    hist = get_histories()
    return render_template(
        'index.html',
        results=None,
        form=None,
        instructions=None,
        instructions_html=None,
        report_file=None,
        blog_post=None,
        blog_html=None,
        blog_file=None,
        logs=None,
        report_prompt='',
        blog_prompt='',
        scrape_history=hist['scrapes'],
        report_history=hist['reports'],
        blog_history=hist['blogs'],
    )


if __name__ == '__main__':
    app.run(port=5007)

