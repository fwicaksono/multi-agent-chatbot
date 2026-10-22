"""Language handling for keyword (full-text) search.

PostgreSQL full-text search needs language rules: a stemmer that reduces words
to their root ("tahunan" -> "tahun", "dibayarkan" -> "bayar") and a list of
stop words to ignore. A knowledge base may mix English and Indonesian
documents, so:

* every chunk is indexed with the rules of its own detected language, and
* queries are matched against all configured languages (see FTS_LANGUAGES).

Detection counts stop words. Function words ("the", "and", "yang", "dan") make
up a large share of any running text and are very distinctive per language,
so this is cheap, deterministic and needs no model. It also covers a gap in
PostgreSQL: its `indonesian` configuration stems words but has no stop-word
list, so "yang" or "dan" in a question would otherwise match nearly every
Indonesian passage. Query terms are therefore filtered with these lists too.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)

STOPWORDS: dict[str, frozenset[str]] = {
    "english": frozenset(
        """
        a about above after again against all also am an and any are as at be because been
        before being below between both but by can could did do does doing down during each
        few for from further had has have having he her here hers him his how if in into is
        it its itself just me more most must my no nor not now of off on once only or other
        our ours out over own per same shall she should so some such than that the their
        theirs them then there these they this those through to too under until up upon very
        was we were what when where which while who whom why will with within would you your
        """.split()
    ),
    "indonesian": frozenset(
        """
        ada adalah agar akan aku anda antara apa apabila apakah atas atau bagaimana bagi
        bahwa banyak beberapa begitu belum berapa berbagai bila bisa boleh dalam dan dapat
        dari demikian dengan di dia hal hanya harus hingga ia ialah ini itu jadi jika juga
        kalau kami kamu karena ke kemudian kenapa kepada ketika kita lagi lain lebih maka
        mana masih masing mengapa mereka meski namun oleh pada para per perlu pula pun
        saat saja sama sampai sangat saya sebagai sebelum sedang sehingga sejak selain
        semua seperti serta setelah setiap siapa sudah supaya tanpa telah tentang tersebut
        tetapi tiap untuk yaitu yakni yang
        """.split()
    ),
}


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def detect_language(text: str, languages: list[str]) -> str:
    """The configured language whose stop words occur most often in `text`.

    Languages without a stop-word list (e.g. `simple`) never win; with no
    signal at all (very short text, tables of numbers) the first configured
    language is used.
    """
    tokens = words(text)
    best, best_hits = languages[0], 0
    for lang in languages:
        stop = STOPWORDS.get(lang)
        if not stop:
            continue
        hits = sum(1 for t in tokens if t in stop)
        if hits > best_hits:
            best, best_hits = lang, hits
    return best


def query_terms(query: str, languages: list[str], limit: int = 32) -> list[str]:
    """Distinct search terms of a question, without stop words of any language.

    If a question consists only of stop words, the words are kept so that it
    still searches for something.
    """
    tokens = list(dict.fromkeys(words(query)))
    stop = frozenset().union(*(STOPWORDS.get(lang, frozenset()) for lang in languages))
    meaningful = [t for t in tokens if t not in stop]
    return (meaningful or tokens)[:limit]
