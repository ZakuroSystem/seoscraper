# seoscraper

Simple CLI tool to search the web and extract article information.

## Usage

Install dependencies (Python 3.12 or compatible is recommended):

```bash
pip install -r requirements.txt
```

Run the scraper with a keyword. Optionally specify the number of results with `-n`:

```bash
python scraper.py "openai" -n 3
```

The tool prints each URL, its domain, the detected publish time and the first 500 characters of paragraph text.
