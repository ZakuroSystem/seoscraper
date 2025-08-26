# seoscraper

キーワードを指定して検索し、記事情報を取得するシンプルな CLI ツールです。
文字化け判定、本文・SEOタイトルの共通部分列分析、ログ出力機能も備えています。

## 特徴

* Google 検索から記事 URL を取得
* HTML を複数エンコード候補でデコードし、文字化けを自動判定
* robots.txt の有無を確認
* 公開日、SEOタイトル、本文の抜粋を取得
* **本文・SEOタイトルの共通部分列分析（共通テキストを上位表示）**
* ログをコンソール・ファイルに出力（ローテーション対応）
* 調査結果を CSV / JSON 形式で保存可能
* Flask を用いた HTML Web UI（ポート5000）に対応
* 出現頻度が高すぎる共通サブ文字列や検索語・短い仮名列を自動除外し分析の信頼性を向上
* 共通サブ文字列ごとの出現割合を表示し、重要度を直感的に評価可能
* tiktoken によるトークン列分析モード（デフォルト）を搭載
* 各ページの語数と上位キーワード頻度を算出
* トークンと文字列を組み合わせたハイブリッド分析で高精度な共通判定
* スレッド並列により検索と取得を高速化（`--workers` で制御、デフォルト10）
* 除外リストに `/正規表現/` を記述して不要なフレーズを柔軟に無視

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
* `--delay` : 各リクエスト前の待機秒数（デフォルト 0.1）
* `--workers` : 並列リクエスト数（デフォルト 10）
* `--chars` : 本文の表示文字数（デフォルト 1000）
* `--log-file` : ログファイルのパスを指定するとファイル出力（省略時はコンソールのみ）
* `--log-level` : ログレベル (`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`)
* `--results-csv` : 結果を CSV に保存
* `--results-json` : 結果とメタ情報を JSON に保存
* `--rank-k` : 共通本文・共通タイトルの上位件数（デフォルト 15）
* `--analyze-chars` : 分析する文字数を指定 (デフォルト5000)
* `--max-common-ratio` : 共通サブ文字列とみなす最大出現率（デフォルト 0.8）
* `--analysis-mode` : 共通判定に用いる解析モード (`tiktoken` / `char` / `hybrid`、デフォルト `tiktoken`)
* `--exclude-file` : 除外文字や `/regex/` を記述したテキストファイル


### 語数・キーワード解析

各ページ本文を単語に分割し、語数と出現頻度上位のキーワードを算出します。CLI / Web UI で表示され、CSV や JSON の結果にも `word_count` と `top_keywords` フィールドとして保存されます。
### Web UI

HTML ベースの Web インターフェースは次のコマンドで起動できます。

```bash
python webui.py
```

ブラウザで [http://localhost:5000](http://localhost:5000) にアクセスしてください。フォームから遅延やワーカー数、解析モード、除外パターンなどすべての主要機能を直感的に設定できます。

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
共通本文サブ文字列:
[2件] 'OpenAI, Inc. は米国の人工知能 (AI) ...'
共通SEOタイトルサブ文字列:
[2件] 'OpenAI - Wikipedia'
```

### ログ出力例（INFOレベル）

```
2025-08-12 15:20:01 INFO: 検索開始 keyword="openai" num=3
2025-08-12 15:20:02 INFO: Search ok: openai (hits=3)
2025-08-12 15:20:05 INFO: OK https://en.wikipedia.org/wiki/OpenAI | title="OpenAI - Wikipedia" robots=True
2025-08-12 15:20:05 INFO: Summary: hits=3, collected=3, skipped=0
2025-08-12 15:20:05 INFO: Top 15 COMMON BODY TEXTS:
2025-08-12 15:20:05 INFO: [BODY 2 / 67%] 'OpenAI, Inc. は米国の人工知能 (AI) ...'
2025-08-12 15:20:05 INFO: Top 15 COMMON SEO TITLE SUBSTRINGS:
2025-08-12 15:20:05 INFO: [TITLE 2] 'OpenAI - Wikipedia'
```
