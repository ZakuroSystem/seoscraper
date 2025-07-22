# seoscraper

キーワードを指定して検索し、記事情報を取得するシンプルな CLI ツールです。

## 使い方

Python 3.12 など互換バージョンで動作します。まず依存パッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

以下のように実行します。`-n` で取得する件数、`--delay` でリクエスト間の待ち時間、`--chars` で本文の表示文字数を変更できます（デフォルト 1000 文字）。robots.txt の有無と SEO タイトルも表示します。

```bash
python scraper.py "openai" -n 3 --delay 1.5 --chars 500
```

実行結果には URL、ドメイン、公開日時、SEO タイトル、robots.txt の有無、本文抜粋が表示されます。
さらに、取得した記事から共通して現れる単語の出現回数も自動集計して最後に一覧表示されます。

例:

```text
URL: https://en.wikipedia.org/wiki/OpenAI
ドメイン: wikipedia.org
公開日: N/A
SEOタイトル: OpenAI - Wikipedia
robots.txt: あり
本文: OpenAI, Inc. は米国の人工知能 (AI) ...
```

共通ワードの例:

```text
openai: 3
ai: 2
research: 2
```
