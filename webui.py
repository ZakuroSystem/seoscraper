import time
import pandas as pd
import streamlit as st

from scraper import (
    create_session,
    get_search_results,
    fetch_html,
    parse_html,
    extract_domain,
    robots_exists,
    common_substrings_rank,
    rank_equal_titles,
)


def run_analysis(keyword: str, num_results: int, delay: float, rank_k: int,
                 analyze_chars: int, max_common_ratio: float, analysis_mode: str):
    """検索と解析を実行し結果を返す"""
    session = create_session()
    urls = get_search_results(keyword, num_results, delay)
    robots_cache = {}
    results = []
    texts = []
    titles = []

    for url in urls:
        html = fetch_html(session, url)
        if not html:
            time.sleep(delay)
            continue
        data = parse_html(html)
        domain = extract_domain(url)
        if domain not in robots_cache:
            robots_cache[domain] = robots_exists(session, url)
        results.append({
            "url": url,
            "domain": domain,
            "published_time": data["published_time"],
            "title": data["title"],
            "robots": robots_cache[domain],
            "text": data["text"][:analyze_chars],
        })
        texts.append(data["text"])
        titles.append(data["title"])
        time.sleep(delay)

    common_subs = common_substrings_rank(
        texts,
        analyze_chars=analyze_chars,
        top_k=rank_k,
        max_doc_ratio=max_common_ratio,
        mode=analysis_mode,
    )
    title_ranks = rank_equal_titles(titles, top_k=rank_k)
    return results, common_subs, title_ranks


def main():
    st.title("SEO Scraper Web UI")
    keyword = st.text_input("Keyword")
    num_results = st.number_input("Number of results", min_value=1, max_value=50, value=10)
    delay = st.number_input("Delay between requests (sec)", min_value=0.0, value=1.0)
    analyze_chars = st.number_input("Analyze characters", min_value=100, max_value=20000, value=5000)
    rank_k = st.number_input("Top K", min_value=1, max_value=50, value=15)
    max_common_ratio = st.slider("Max common substring ratio", min_value=0.5, max_value=1.0, value=0.8)
    analysis_mode = st.selectbox("Analysis mode", ["tiktoken", "char"], index=0)

    if st.button("Run") and keyword:
        with st.spinner("Scraping..."):
            results, common_subs, title_ranks = run_analysis(
                keyword,
                int(num_results),
                float(delay),
                int(rank_k),
                int(analyze_chars),
                float(max_common_ratio),
                analysis_mode,
            )
        if results:
            st.subheader("Results")
            st.dataframe(pd.DataFrame(results))

            st.subheader("Common substrings")
            for sub, cnt in common_subs:
                st.write(f"{cnt} docs: {sub}")

            st.subheader("Equal SEO titles")
            for title, cnt in title_ranks:
                st.write(f"{cnt} docs: {title}")
        else:
            st.write("No results found")


if __name__ == "__main__":
    main()
