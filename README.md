# seoscraper

キーワードを指定して検索し、記事情報を取得するシンプルな CLI ツールです。
文字化け判定、完全一致による本文・SEOタイトルの共通判定、ログ出力機能も備えています。

## 特徴

* Google 検索から記事 URL を取得
* HTML を複数エンコード候補でデコードし、文字化けを自動判定
* robots.txt の有無を確認
* 公開日、SEOタイトル、本文の抜粋を取得
* **本文・SEOタイトルの完全一致集計（共通テキストを上位表示）**
* ログをコンソール・ファイルに出力（ローテーション対応）
* 調査結果を CSV / JSON 形式で保存可能
* Streamlit ベースのリッチな Web UI に対応
* 出現頻度が高すぎる共通サブ文字列を自動除外し分析の信頼性を向上

## インストール

Python 3.12 など互換バージョンで動作します。まず依存パッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

## 使い方

基本的な実行例:

```bash
python scraper.py "openai" -n 3 --delay 1.5 --chars 500
```

* `-n` : 取得件数（デフォルト 10）
* `--delay` : リクエスト間隔（秒）
* `--chars` : 本文の表示文字数（デフォルト 1000）
* `--log-file` : ログファイルのパスを指定するとファイル出力（省略時はコンソールのみ）
* `--log-level` : ログレベル (`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`)
* `--results-csv` : 結果を CSV に保存
* `--results-json` : 結果とメタ情報を JSON に保存
* `--rank-k` : 共通本文・共通タイトルの上位件数（デフォルト 15）
* `--analyze-chars` : 分析する文字数を指定 (デフォルト5000)
* `--max-common-ratio` : 共通サブ文字列とみなす最大出現率（デフォルト 0.8）

### Web UI

リッチな Web インターフェースは次のコマンドで起動できます。

```bash
streamlit run webui.py
```

### 実行例

```bash
python scraper.py "openai" -n 3 --delay 1.5 --chars 500 \
  --log-file scrape.log --log-level INFO \
  --results-csv results.csv --results-json results.json \
  --rank-k 10 --analyze-chars 4000 --max-common-ratio 0.8
```

### 出力例

```text
URL: https://en.wikipedia.org/wiki/OpenAI
ドメイン: wikipedia.org
公開日: N/A
SEOタイトル: OpenAI - Wikipedia
robots.txt: あり
本文: OpenAI, Inc. は米国の人工知能 (AI) ...
--------------------------------------------------------------------------------
共通本文（完全一致）:
[2件] 'OpenAI, Inc. は米国の人工知能 (AI) ...'
共通SEOタイトル（完全一致）:
[2件] 'OpenAI - Wikipedia'
```

### ログ出力例（INFOレベル）

```
2025-08-12 15:20:01 INFO: 検索開始 keyword="openai" num=3
2025-08-12 15:20:02 INFO: Search ok: openai (hits=3)
2025-08-12 15:20:05 INFO: OK https://en.wikipedia.org/wiki/OpenAI | title="OpenAI - Wikipedia" robots=True
2025-08-12 15:20:05 INFO: Summary: hits=3, collected=3, skipped=0
2025-08-12 15:20:05 INFO: Top 15 COMMON BODY TEXTS:
2025-08-12 15:20:05 INFO: [BODY 2] 'OpenAI, Inc. は米国の人工知能 (AI) ...'
2025-08-12 15:20:05 INFO: Top 15 COMMON SEO TITLES:
2025-08-12 15:20:05 INFO: [TITLE 2] 'OpenAI - Wikipedia'
```
