from typing import Dict
from flask import Flask, render_template, request, url_for, Response, stream_with_context
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import markdown
import json

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
    generate_similar_keywords,
    ollama_chat_stream,
    save_blog_markdown,
    markdown_to_html_ai,
    save_report_markdown,
    save_scrape_json,
    have_ollama_model,
    DEFAULT_EXTRA_INSTRUCTION,
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
    body_match: bool,
    report_prompt: str = "",
    blog_prompt: str = "",
    blog_style: str = "",
    human_mode: bool = False,
    model: str = "gpt-oss:20b",
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
            "headings": data.get("headings", []),
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
    common_subs = []
    if body_match:
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
        if have_ollama_model(model):
            with logs_lock:
                logs.append("Ollamaで指示書生成をリクエストしています")
            instructions, err = generate_blog_instruction(
                keyword, results, common_subs, title_ranks, report_prompt,
                model=model,
                timeout=690 if model == "gpt-oss:120b" else 160,
            )
            if instructions:
                with logs_lock:
                    logs.append("指示書を生成しました")
            else:
                with logs_lock:
                    logs.append(f"指示書生成失敗: {err}")
        else:
            with logs_lock:
                logs.append(f"{model}が見つからないため指示書生成をスキップしました")
    else:
        with logs_lock:
            logs.append("指示書生成をスキップしました")
    blog_post = None
    blog_file = None
    if generate_blog:
        if instructions:
            if have_ollama_model(model):
                try:
                    with logs_lock:
                        logs.append("Ollamaでブログ生成をリクエストしています")
                    blog_post, err = generate_blog_post(
                        keyword,
                        instructions,
                        blog_prompt,
                        style=blog_style,
                        human_mode=human_mode,
                        model=model,
                        timeout=690 if model == "gpt-oss:120b" else 160,
                    )
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
                    logs.append(f"{model}が見つからないためブログ生成をスキップしました")
        else:
            with logs_lock:
                logs.append("指示書がないためブログ生成をスキップしました")
    else:
        with logs_lock:
            logs.append("ブログ生成をスキップしました")
    return results, common_subs, title_ranks, instructions, blog_post, blog_file, logs


@app.route('/stream_report')
def stream_report():
    if not last_state.get('results'):
        def gen_empty():
            yield "data: {\"error\": \"no results\"}\n\n"
        return Response(gen_empty(), mimetype='text/event-stream')
    prompt = request.args.get('prompt', '')
    hi = request.args.get('hi') == '1'
    model = 'gpt-oss:120b' if hi else 'gpt-oss:20b'
    timeout = 690 if hi else 160
    if not have_ollama_model(model):
        def gen_model():
            yield f"data: {{\"error\": \"{model} not available\"}}\n\n"
        return Response(gen_model(), mimetype='text/event-stream')

    results = last_state['results']
    common_subs = last_state['common_subs']
    title_ranks = last_state['title_ranks']
    keyword = last_state['keyword']
    summary_lines = []
    for r in results[:5]:
        kws = ", ".join(k["keyword"] for k in r.get("top_keywords", [])[:3])
        heads = "/".join(r.get("headings", [])[:3])
        summary_lines.append(f"- {r.get('title', '')} | 見出し: {heads} | キーワード: {kws}")
    body = "\n".join(summary_lines)
    subs = "\n".join(f"- {s['text']} ({s['count']}件)" for s in common_subs[:5])
    titles = "\n".join(f"- {t} ({c}件)" for t, c in title_ranks[:5])
    prompt_txt = (
        f"検索キーワード: {keyword}\n"
        f"上位ページの概要:\n{body}\n\n"
    )
    if subs:
        prompt_txt += f"共通本文フレーズ:\n{subs}\n\n"
    prompt_txt += (
        f"共通SEOタイトルフレーズ:\n{titles}\n\n"
        "これらを参考に、どのようなブログ記事を書けばよいかを日本語でまとめた指示書を作成してください。"
    )
    if prompt:
        prompt_txt += f"\n\n追加指示:\n{prompt}"
    messages = [
        {"role": "system", "content": "You are an expert Japanese SEO consultant."},
        {"role": "user", "content": prompt_txt},
    ]

    def generate():
        yield f"data: {{\"status\": \"指示書生成を開始します\"}}\n\n"
        buf = []
        for token in ollama_chat_stream(model, messages, timeout=timeout):
            buf.append(token)
            yield f"data: {{\"token\": {json.dumps(token)} }}\n\n"
        full = ''.join(buf)
        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'reports')
        path = save_report_markdown(full, keyword, directory=static_dir)
        instructions_html = markdown.markdown(full, extensions=["extra"])
        last_state.update({
            'instructions': full,
            'instructions_html': instructions_html,
            'report_file': os.path.basename(path),
            'report_prompt': prompt,
        })
        done_payload = {
            "done": True,
            "html": instructions_html,
            "file": os.path.basename(path),
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/stream_blog')
def stream_blog():
    if not last_state.get('instructions'):
        def gen_empty():
            yield "data: {\"error\": \"no instructions\"}\n\n"
        return Response(gen_empty(), mimetype='text/event-stream')
    prompt = request.args.get('prompt', '')
    style = request.args.get('style', '')
    human = request.args.get('human') == '1'
    info = request.args.get('info') == '1'
    hi = request.args.get('hi') == '1'
    model = 'gpt-oss:120b' if hi else 'gpt-oss:20b'
    timeout = 690 if hi else 160
    if not have_ollama_model(model):
        def gen_model():
            yield f"data: {{\"error\": \"{model} not available\"}}\n\n"
        return Response(gen_model(), mimetype='text/event-stream')

    keyword = last_state['keyword']
    instructions = last_state['instructions']
    fmt = "Markdown"
    prompt_txt = (
        f"検索キーワード: {keyword}\n"
        f"以下の指示書に従って、日本語でSEOに最適化されたブログ記事を{fmt}形式で作成してください。\n\n"
        f"{instructions}\n"
    )
    if style:
        prompt_txt += f"\nブログの雰囲気: {style}\n"
    if human:
        prompt_txt += (
            "\nAIで生成されたと気付かれないよう、人間らしい自然な語り口で書いてください。"
            "以下の点にも留意してください:\n"
            "- 見出し番号など機械的な構成を避け、節ごとに変化を付ける\n"
            "- 金額や数値は現実的な揺らぎを含め、過度に整列させない\n"
            "- 不自然な語句や誤用を避け、文脈に合う語彙を選ぶ\n"
            "- 汎用的なマーケティング文句を多用せず具体例や根拠を示す\n"
            "- 表やFAQなどの装飾は必要なものだけに留める\n"
        )
    if prompt:
        prompt_txt += f"\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}\n{prompt}\n"
    else:
        prompt_txt += f"\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}\n"
    if info:
        prompt_txt += "\n案件や見積りへの誘導は行わず、情報提供のみに集中してください。"
    messages = [
        {
            "role": "system",
            "content": f"You are a skilled Japanese blogger. Output {fmt}.",
        },
        {"role": "user", "content": prompt_txt},
    ]

    def generate():
        yield f"data: {{\"status\": \"ブログ生成を開始します\"}}\n\n"
        buf = []
        for token in ollama_chat_stream(model, messages, timeout=timeout):
            buf.append(token)
            yield f"data: {{\"token\": {json.dumps(token)} }}\n\n"
        full = ''.join(buf)
        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'blogs')
        path = save_blog_markdown(full, keyword, directory=static_dir)
        blog_html = markdown.markdown(full, extensions=["extra"])
        if info:
            last_state.update({
                'info_blog_post': full,
                'info_blog_html': blog_html,
                'info_blog_file': os.path.basename(path),
                'blog_prompt': prompt,
                'blog_style': style,
                'human_mode': human,
                'info_html_mode': False,
            })
        else:
            last_state.update({
                'blog_post': full,
                'blog_html': blog_html,
                'blog_file': os.path.basename(path),
                'blog_prompt': prompt,
                'blog_style': style,
                'human_mode': human,
                'html_mode': False,
            })
        done_payload = {
            "done": True,
            "html": blog_html,
            "file": os.path.basename(path),
            "html_mode": False,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/', methods=['GET', 'POST'])
def index():
    global last_state
    if request.method == 'POST':
        action = request.form.get('action', 'scrape')
        if action == 'expand':
            keyword = request.form.get('keyword', '')
            logs = []
            if keyword:
                extra, err = generate_similar_keywords(keyword)
                if extra:
                    keyword = keyword + ' ' + ' '.join(extra)
                    logs.append('類似キーワードを追加: ' + ', '.join(extra))
                else:
                    logs.append(f'類似キーワード生成失敗: {err}')
            form = request.form.to_dict(flat=True)
            form['keyword'] = keyword
            hist = get_histories()
            return render_template(
                'index.html',
                results=None,
                logs=logs,
                form=form,
                instructions=None,
                instructions_html=None,
                report_file=None,
                blog_post=None,
                blog_html=None,
                blog_file=None,
                info_blog_post=None,
                info_blog_html=None,
                info_blog_file=None,
                info_html_mode=False,
                report_prompt='',
                blog_prompt='',
                blog_style='標準',
                human_mode=False,
                html_mode=False,
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
        if action == 'scrape':
            keyword = request.form.get('keyword', '')
            if keyword:
                num_results = min(int(request.form.get('num_results', 10)), 50)
                delay = float(request.form.get('delay', 0.1))
                workers = int(request.form.get('workers', 10))
                analyze_chars = int(request.form.get('analyze_chars', 5000))
                rank_k = int(request.form.get('rank_k', 15))
                max_common_ratio = float(request.form.get('max_common_ratio', 0.8))
                analysis_mode = request.form.get('analysis_mode', 'tiktoken')
                merge_percent = float(request.form.get('merge_percent', 18.0))
                excl_lines = request.form.get('exclude_patterns', '').splitlines()
                _, remove_trans, remove_patterns = parse_exclude_lines(excl_lines)
                skip_common = bool(request.form.get('skip_common'))
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
                    body_match=not skip_common,
                    model="gpt-oss:20b",
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
                    'skip_common': skip_common,
                    'instructions': None,
                    'instructions_html': None,
                    'report_file': None,
                    'blog_post': None,
                    'blog_html': None,
                    'blog_file': None,
                    'info_blog_post': None,
                    'info_blog_html': None,
                    'info_blog_file': None,
                    'info_html_mode': False,
                    'report_prompt': '',
                    'blog_prompt': '',
                    'blog_style': '標準',
                    'human_mode': False,
                    'html_mode': False,
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
                info_blog_post=None,
                info_blog_html=None,
                info_blog_file=None,
                info_html_mode=False,
                logs=logs,
                form=request.form,
                report_prompt='',
                blog_prompt='',
                blog_style='標準',
                    human_mode=False,
                    html_mode=False,
                    scrape_history=hist['scrapes'],
                    report_history=hist['reports'],
                    blog_history=hist['blogs'],
                )
        elif action == 'report' and last_state.get('results'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            report_prompt = request.form.get('report_prompt', '')
            hi_model = bool(request.form.get('hi_model'))
            model = "gpt-oss:120b" if hi_model else "gpt-oss:20b"
            timeout = 690 if hi_model else 160
            if have_ollama_model(model):
                logs.append("Ollamaで指示書生成をリクエストしています")
                instructions, err = generate_blog_instruction(
                    keyword,
                    last_state['results'],
                    last_state['common_subs'],
                    last_state['title_ranks'],
                    report_prompt,
                    model=model,
                    timeout=timeout,
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
                logs.append(f"{model}が見つからないため指示書生成をスキップしました")
            instructions_html = (
                markdown.markdown(instructions, extensions=["extra"])
                if instructions
                else None
            )
            last_state.update({
                'instructions': instructions,
                'instructions_html': instructions_html,
                'report_file': report_file,
                'logs': logs,
                'report_prompt': report_prompt,
            })
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
                info_blog_post=last_state.get('info_blog_post'),
                info_blog_html=last_state.get('info_blog_html'),
                info_blog_file=last_state.get('info_blog_file'),
                logs=logs,
                form=last_state.get('form'),
                report_prompt=report_prompt,
                blog_prompt='',
                blog_style=last_state.get('blog_style', '標準'),
                human_mode=last_state.get('human_mode', False),
                html_mode=last_state.get('html_mode', False),
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
        elif action == 'blog' and last_state.get('instructions'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            blog_prompt = request.form.get('blog_prompt', '')
            blog_style = request.form.get('blog_style', '標準')
            human_mode = bool(request.form.get('human_mode'))
            hi_model = bool(request.form.get('hi_model'))
            model = "gpt-oss:120b" if hi_model else "gpt-oss:20b"
            timeout = 690 if hi_model else 160
            if have_ollama_model(model):
                logs.append("Ollamaでブログ生成をリクエストしています")
                blog_post, err = generate_blog_post(
                    keyword,
                    last_state['instructions'],
                    blog_prompt,
                    style=blog_style,
                    human_mode=human_mode,
                    model=model,
                    timeout=timeout,
                )
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
                logs.append(f"{model}が見つからないためブログ生成をスキップしました")
            blog_html = markdown.markdown(blog_post, extensions=["extra"]) if blog_post else None
            last_state.update({
                'blog_post': blog_post,
                'blog_html': blog_html,
                'blog_file': blog_file,
                'logs': logs,
                'blog_prompt': blog_prompt,
                'blog_style': blog_style,
                'human_mode': human_mode,
                'html_mode': False,
            })
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
                info_blog_post=last_state.get('info_blog_post'),
                info_blog_html=last_state.get('info_blog_html'),
                info_blog_file=last_state.get('info_blog_file'),
                info_html_mode=last_state.get('info_html_mode', False),
                logs=logs,
                form=last_state.get('form'),
                report_prompt=last_state.get('report_prompt', ''),
                blog_prompt=blog_prompt,
                blog_style=blog_style,
                human_mode=human_mode,
                html_mode=False,
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
        elif action == 'convert_html' and last_state.get('blog_post'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            html, err = markdown_to_html_ai(last_state['blog_post'])
            if html:
                static_dir = os.path.join(os.path.dirname(__file__), 'static', 'blogs')
                path = save_blog_markdown(html, keyword, directory=static_dir, html=True)
                blog_file = os.path.basename(path)
                logs.append("MarkdownをHTMLに変換しました")
                last_state.update({'blog_html': html, 'blog_file': blog_file, 'html_mode': True, 'logs': logs})
            else:
                blog_file = last_state.get('blog_file')
                logs.append(f"HTML化失敗: {err}")
                last_state.update({'logs': logs})
                hist = get_histories()
                return render_template(
                    'index.html',
                    results=last_state['results'],
                    common_subs=last_state['common_subs'],
                    title_ranks=last_state['title_ranks'],
                    instructions=last_state.get('instructions'),
                    instructions_html=last_state.get('instructions_html'),
                    report_file=last_state.get('report_file'),
                    blog_post=last_state.get('blog_post'),
                    blog_html=last_state.get('blog_html'),
                    blog_file=blog_file,
                    info_blog_post=last_state.get('info_blog_post'),
                    info_blog_html=last_state.get('info_blog_html'),
                    info_blog_file=last_state.get('info_blog_file'),
                    info_html_mode=last_state.get('info_html_mode', False),
                    logs=logs,
                    form=last_state.get('form'),
                    report_prompt=last_state.get('report_prompt', ''),
                    blog_prompt=last_state.get('blog_prompt', ''),
                    blog_style=last_state.get('blog_style', '標準'),
                    human_mode=last_state.get('human_mode', False),
                    html_mode=last_state.get('html_mode', False),
                    scrape_history=hist['scrapes'],
                    report_history=hist['reports'],
                    blog_history=hist['blogs'],
                )
            hist = get_histories()
            return render_template(
                'index.html',
                results=last_state['results'],
                common_subs=last_state['common_subs'],
                title_ranks=last_state['title_ranks'],
                instructions=last_state.get('instructions'),
                instructions_html=last_state.get('instructions_html'),
                report_file=last_state.get('report_file'),
                blog_post=last_state.get('blog_post'),
                blog_html=html,
                blog_file=blog_file,
                info_blog_post=last_state.get('info_blog_post'),
                info_blog_html=last_state.get('info_blog_html'),
                info_blog_file=last_state.get('info_blog_file'),
                info_html_mode=last_state.get('info_html_mode', False),
                logs=logs,
                form=last_state.get('form'),
                report_prompt=last_state.get('report_prompt', ''),
                blog_prompt=last_state.get('blog_prompt', ''),
                blog_style=last_state.get('blog_style', '標準'),
                human_mode=last_state.get('human_mode', False),
                html_mode=True,
                scrape_history=hist['scrapes'],
                report_history=hist['reports'],
                blog_history=hist['blogs'],
            )
        elif action == 'blog_info' and last_state.get('results'):
            logs = last_state.get('logs', []).copy()
            keyword = last_state['keyword']
            blog_prompt = request.form.get('blog_prompt', '')
            blog_style = request.form.get('blog_style', '標準')
            human_mode = bool(request.form.get('human_mode'))
            hi_model = bool(request.form.get('hi_model'))
            model = "gpt-oss:120b" if hi_model else "gpt-oss:20b"
            timeout = 690 if hi_model else 160
            if have_ollama_model(model):
                logs.append("Ollamaで指示書生成をリクエストしています(情報提供)")
                instructions, err = generate_blog_instruction(
                    keyword,
                    last_state['results'],
                    last_state['common_subs'],
                    last_state['title_ranks'],
                    last_state.get('report_prompt', ''),
                    info_only=True,
                    model=model,
                    timeout=timeout,
                )
                if instructions:
                    logs.append("指示書を生成しました")
                    logs.append("Ollamaでブログ生成をリクエストしています(情報提供)")
                    blog_post, err = generate_blog_post(
                        keyword,
                        instructions,
                        blog_prompt,
                        style=blog_style,
                        human_mode=human_mode,
                        info_only=True,
                        model=model,
                        timeout=timeout,
                    )
                    if blog_post:
                        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'blogs')
                        path = save_blog_markdown(blog_post, keyword, directory=static_dir)
                        blog_file = os.path.basename(path)
                        logs.append("情報提供ブログ記事を保存しました")
                    else:
                        blog_file = None
                        logs.append(f"ブログ生成失敗: {err}")
                else:
                    blog_post = None
                    blog_file = None
                    logs.append(f"指示書生成失敗: {err}")
            else:
                blog_post = None
                blog_file = None
                logs.append(f"{model}が見つからないためブログ生成をスキップしました")
            blog_html = markdown.markdown(blog_post, extensions=["extra"]) if blog_post else None
            last_state.update({
                'info_blog_post': blog_post,
                'info_blog_html': blog_html,
                'info_blog_file': blog_file,
                'logs': logs,
                'blog_prompt': blog_prompt,
                'blog_style': blog_style,
                'human_mode': human_mode,
                'info_html_mode': False,
            })
            hist = get_histories()
            return render_template(
                'index.html',
                results=last_state['results'],
                common_subs=last_state['common_subs'],
                title_ranks=last_state['title_ranks'],
                instructions=last_state.get('instructions'),
                instructions_html=last_state.get('instructions_html'),
                report_file=last_state.get('report_file'),
                blog_post=last_state.get('blog_post'),
                blog_html=last_state.get('blog_html'),
                blog_file=last_state.get('blog_file'),
                info_blog_post=blog_post,
                info_blog_html=blog_html,
                info_blog_file=blog_file,
                info_html_mode=last_state.get('info_html_mode', False),
                logs=logs,
                form=last_state.get('form'),
                report_prompt=last_state.get('report_prompt', ''),
                blog_prompt=blog_prompt,
                blog_style=blog_style,
                human_mode=human_mode,
                html_mode=last_state.get('html_mode', False),
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
        info_blog_post=None,
        info_blog_html=None,
        info_blog_file=None,
        info_html_mode=False,
        logs=None,
        report_prompt='',
        blog_prompt='',
        blog_style='標準',
        human_mode=False,
        html_mode=False,
        scrape_history=hist['scrapes'],
        report_history=hist['reports'],
        blog_history=hist['blogs'],
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)

