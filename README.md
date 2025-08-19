# seoscraper

Utility helpers for working with Japanese text and SEO titles.

## Features

- `encode_text`/`decode_tokens` wrap [tiktoken](https://github.com/openai/tiktoken)
  and normalise text so that mojibake like `�` does not appear.
- `seo_title_similarity` performs fuzzy matching between an SEO title and
  article text instead of relying on an exact match.

## Testing

```bash
pytest
```
