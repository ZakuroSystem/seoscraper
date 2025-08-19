from text_utils import encode_text, decode_tokens, seo_title_similarity


def test_round_trip_japanese_text():
    text = "兵庫県尼崎市"
    tokens = encode_text(text)
    decoded = decode_tokens(tokens)
    assert decoded == text


def test_encode_decode_removes_replacement_char():
    garbled = "�庫県尼崎市"
    tokens = encode_text(garbled)
    decoded = decode_tokens(tokens)
    assert "�" not in decoded


def test_seo_title_similarity_fuzzy():
    title = "兵庫県尼崎市での作業"
    text = "兵庫県尼崎市の作業所を紹介します。"
    score = seo_title_similarity(title, text)
    assert 0.5 < score < 1.0
