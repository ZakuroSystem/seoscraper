from typing import List, Dict, Tuple


# 共通追加指示
DEFAULT_EXTRA_INSTRUCTION = (
    "CTA（コール・トゥ・アクション）の要素を含めないようにしてください。\n"
    "見出しをコピーしないで、ユニークな見出しを作ってください。"
)


def build_instruction_messages(
    keyword: str,
    results: List[Dict],
    common_subs: List[Dict],
    title_ranks: List[Tuple[str, int]],
    user_prompt: str = "",
    info_only: bool = False,
) -> List[Dict[str, str]]:
    """検索結果から指示書生成用のメッセージを組み立てる。"""
    summary_lines = []
    for r in results[:5]:
        kws = ", ".join(k["keyword"] for k in r.get("top_keywords", [])[:3])
        heads = "/".join(r.get("headings", [])[:3])
        summary_lines.append(f"- {r.get('title', '')} | 見出し: {heads} | キーワード: {kws}")
    body = "\n".join(summary_lines)
    subs = "\n".join(f"- {s['text']} ({s['count']}件)" for s in common_subs[:5])
    titles = "\n".join(f"- {t} ({c}件)" for t, c in title_ranks[:5])
    prompt = (
        f"検索キーワード: {keyword}\n"
        f"上位ページの概要:\n{body}\n\n"
    )
    if subs:
        prompt += f"共通本文フレーズ:\n{subs}\n\n"
    prompt += (
        f"共通SEOタイトルフレーズ:\n{titles}\n\n"
        "これらを参考に、どのようなブログ記事を書けばよいかを日本語でまとめた指示書を作成してください。"
    )
    if info_only:
        prompt += "案件や見積りへの誘導は避け、純粋な情報提供に徹してください。"
    if user_prompt:
        prompt += f"\n\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}\n{user_prompt}"
    else:
        prompt += f"\n\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}"
    return [
        {"role": "system", "content": "You are an expert Japanese SEO consultant."},
        {"role": "user", "content": prompt},
    ]


def build_blog_messages(
    keyword: str,
    instructions: str,
    user_prompt: str = "",
    style: str = "",
    human_mode: bool = False,
    info_only: bool = False,
    html_mode: bool = False,
) -> List[Dict[str, str]]:
    """ブログ本文生成用のメッセージを組み立てる。"""
    fmt = "HTML" if html_mode else "Markdown"
    prompt = (
        f"検索キーワード: {keyword}\n"
        f"以下の指示書に従って、日本語でSEOに最適化されたブログ記事を{fmt}形式で作成してください。\n\n"
        f"{instructions}\n"
    )
    if style:
        prompt += f"\nブログの雰囲気: {style}\n"
    if human_mode:
        prompt += (
            "\nAIで生成されたと気付かれないよう、人間らしい自然な語り口で書いてください。"
            "以下の点にも留意してください:\n"
            "- 見出し番号など機械的な構成を避け、節ごとに変化を付ける\n"
            "- 金額や数値は現実的な揺らぎを含め、過度に整列させない\n"
            "- 不自然な語句や誤用を避け、文脈に合う語彙を選ぶ\n"
            "- 汎用的なマーケティング文句を多用せず具体例や根拠を示す\n"
            "- 表やFAQなどの装飾は必要なものだけに留める\n"
        )
    if user_prompt:
        prompt += f"\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}\n{user_prompt}\n"
    else:
        prompt += f"\n追加指示:\n{DEFAULT_EXTRA_INSTRUCTION}\n"
    if info_only:
        prompt += "\n案件や見積りへの誘導は行わず、情報提供のみに集中してください。"
    return [
        {
            "role": "system",
            "content": f"You are a skilled Japanese blogger. Output {fmt}.",
        },
        {"role": "user", "content": prompt},
    ]


def build_html_messages(md: str) -> List[Dict[str, str]]:
    prompt = (
        "次のMarkdownブログ記事をHTMLに変換してください。"
        "色や太字などを用いて読みやすい書式にしてください。"
        f"{DEFAULT_EXTRA_INSTRUCTION}\n\n{md}"
    )
    return [
        {"role": "system", "content": "You are an HTML formatter."},
        {"role": "user", "content": prompt},
    ]


def build_review_messages(blog: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are an expert Japanese editor."},
        {"role": "user", "content": f"このブログを評価して修正点をまとめてください。\n\n{blog}"},
    ]


def build_revise_messages(blog: str, review: str, style: str = "", human: bool = False) -> List[Dict[str, str]]:
    prompt = (
        "ブログを修正点を適応して読みやすく修正してください。\n\n"
        f"修正点:\n{review}\n\nブログ本文:\n{blog}"
    )
    if style:
        prompt += f"\nブログの雰囲気: {style}"
    if human:
        prompt += "\n人間らしい自然な語り口で書いてください。"
    return [
        {"role": "system", "content": "You are a skilled Japanese blogger. Output Markdown."},
        {"role": "user", "content": prompt},
    ]
