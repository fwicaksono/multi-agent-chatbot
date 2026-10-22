from app.ingestion.language import detect_language, query_terms

LANGS = ["english", "indonesian"]


def test_detects_english_and_indonesian_text():
    english = "Permanent employees are entitled to 18 working days of paid annual leave per year."
    indonesian = (
        "Karyawan yang telah bekerja selama 12 bulan berhak atas cuti tahunan dengan upah penuh."
    )
    assert detect_language(english, LANGS) == "english"
    assert detect_language(indonesian, LANGS) == "indonesian"


def test_no_signal_falls_back_to_first_configured_language():
    assert detect_language("IDR 150,000,000 | 2026", LANGS) == "english"
    assert detect_language("IDR 150,000,000 | 2026", ["indonesian", "english"]) == "indonesian"


def test_languages_without_stop_words_never_win():
    assert detect_language("the leave policy of the company", ["simple", "english"]) == "english"
    assert detect_language("anything", ["simple"]) == "simple"


def test_query_terms_drop_stop_words_of_every_language():
    assert query_terms("Berapa hari cuti yang saya dapat?", LANGS) == ["hari", "cuti"]
    assert query_terms("What is the notice period?", LANGS) == ["notice", "period"]


def test_query_of_only_stop_words_still_searches():
    assert query_terms("apa itu?", LANGS) == ["apa", "itu"]
