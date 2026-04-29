"""
Parse tikkun-data.js and tikkun-search-corpus.js from the etnachta subrepo
into Python dicts. Also provides utilities for extracting teamim from Hebrew text.
"""

import json
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

_ROOT = Path(__file__).parent
# Corpus JS files live in assets/ (tracked). Fall back to etnachta/assets/ for
# local dev where the submodule is checked out.
ETNACHTA_DIR = _ROOT / "assets" if (_ROOT / "assets" / "tikkun-data.js").exists() \
    else _ROOT / "etnachta" / "assets"
TIKKUN_DATA_JS   = ETNACHTA_DIR / "tikkun-data.js"
SEARCH_CORPUS_JS = ETNACHTA_DIR / "tikkun-search-corpus.js"

# ---------------------------------------------------------------------------
# Unicode ranges for cantillation marks
# ---------------------------------------------------------------------------
# Primary teamim: U+0591-U+05AF
# Also U+05BD (meteg/silluq), U+05C0 (paseq), U+05C3 (sof pasuq)
TAAM_RANGE_RE = re.compile(r'[\u0591-\u05AF\u05BD\u05C0\u05C3]')

# Mapping from Unicode code point to canonical taam name
UNICODE_TO_TAAM = {
    '\u0591': 'etnahta',        # ֑  (= etnachta)
    '\u0592': 'segolta',        # ֒
    '\u0593': 'shalshelet',     # ֓
    '\u0594': 'zaqef-qatan',    # ֔
    '\u0595': 'zaqef-gadol',    # ֕
    '\u0596': 'tipcha',         # ֖  tipeha
    '\u0597': 'revii',          # ֗
    '\u0598': 'zarqa',          # ֘
    '\u0599': 'pashta',         # ֙
    '\u059A': 'yetiv',          # ֚
    '\u059B': 'tevir',          # ֛
    '\u059C': 'geresh',         # ֜
    '\u059D': 'geresh',         # ֝  geresh muqdam (treat as geresh)
    '\u059E': 'gershayim',      # ֞
    '\u059F': 'qarney-farah',   # ֟  qarney para
    '\u05A0': 'telisha-gedola', # ֠
    '\u05A1': 'pazer',          # ֡
    '\u05A2': 'atnah-hafukh',   # ֢  not used in WLC; munah-legarmeh = munah + paseq
    '\u05A3': 'munah',          # ֣
    '\u05A4': 'mahapakh',       # ֤
    '\u05A5': 'merkha',         # ֥
    '\u05A6': 'merkha-kefula',  # ֦
    '\u05A7': 'darga',          # ֧
    '\u05A8': 'qadma',          # ֨
    '\u05A9': 'telisha-qetana', # ֩
    '\u05AA': 'yerah-ben-yomo', # ֪
    '\u05AB': 'meayla',         # ֫  ole
    '\u05AC': 'meayla',         # ֬  iluy (treat as meayla variant)
    '\u05AD': 'metiga',         # ֭  dehi
    '\u05AE': 'zinor',          # ֮
    # U+05BD (meteg/ga'ya) intentionally omitted — not a taam.
    # Silluq is captured via U+05C3 on the same final word.
    '\u05C0': 'paseq',          # ׀
    '\u05C3': 'silluq',         # ׃  sof pasuq — marks the silluq word
}

TAAM_TO_UNICODE = {v: k for k, v in UNICODE_TO_TAAM.items()}
# etnachta alias
UNICODE_TO_TAAM['\u0591'] = 'etnachta'

# ---------------------------------------------------------------------------
# JS data extraction
# ---------------------------------------------------------------------------

def _extract_json_from_js(path: Path, var_name: str) -> dict:
    """Strip the JS assignment wrapper and return parsed JSON."""
    text = path.read_text(encoding="utf-8")
    # e.g. window.SHIRA_TIKKUN_DATA={...}  or  window.SHIRA_TIKKUN_SEARCH_BUNDLE={...}
    idx = text.find("{")
    if idx == -1:
        raise ValueError(f"No JSON object found in {path}")
    # Find matching closing brace
    depth = 0
    end = idx
    for i, ch in enumerate(text[idx:], start=idx):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(text[idx:end + 1])


@lru_cache(maxsize=1)
def load_tikkun_data() -> dict:
    """
    Returns {book: {chapter: {verse: text_with_teamim}}} for all available books.
    Merges tikkun-data.js (Torah + 1SA) with tikkun-search-corpus.js (all other books).
    tikkun-data.js takes precedence where books overlap.
    """
    primary = _extract_json_from_js(TIKKUN_DATA_JS, "SHIRA_TIKKUN_DATA")["books"]
    secondary = _extract_json_from_js(SEARCH_CORPUS_JS, "SHIRA_TIKKUN_SEARCH_BUNDLE")["books"]
    merged = {**secondary, **primary}   # primary wins on overlap
    return merged


@lru_cache(maxsize=1)
def available_books() -> list[str]:
    return sorted(load_tikkun_data().keys())


@lru_cache(maxsize=1)
def book_labels() -> dict[str, dict]:
    data = _extract_json_from_js(TIKKUN_DATA_JS, "SHIRA_TIKKUN_DATA")
    return data.get("bookLabels", {})


# ---------------------------------------------------------------------------
# Taam extraction utilities
# ---------------------------------------------------------------------------

# Hebrew base-letter range (excludes nikud and teamim)
HEBREW_LETTER_RE = re.compile(r'[\u05D0-\u05EA\u05F0-\u05F4\uFB1D-\uFB4E]')
# Nikud (vowel points) to strip when getting "plain" text
NIKUD_RE = re.compile(r'[\u05B0-\u05BC\u05C1\u05C2\u05C4\u05C5\u05C7]')
# Maqaf (maqqef) connector
MAQAF = '\u05BE'


def strip_teamim(text: str) -> str:
    """Remove cantillation marks, leaving letters + nikud."""
    return TAAM_RANGE_RE.sub('', text)


def strip_nikud_and_teamim(text: str) -> str:
    """Remove all diacritics, leaving bare Hebrew letters."""
    return NIKUD_RE.sub('', strip_teamim(text))


def word_teamim(word: str) -> list[str]:
    """Return the raw list of taam names on a single word string (no context)."""
    found = []
    seen = set()
    for ch in word:
        if ch in UNICODE_TO_TAAM:
            name = UNICODE_TO_TAAM[ch]
            if name not in seen:
                seen.add(name)
                found.append(name)
    return found


def verse_word_list(verse_text: str) -> list[dict]:
    """
    Split a verse into a list of word dicts:
      { 'text': str,            # word with nikud + teamim
        'plain': str,           # bare Hebrew letters only
        'teamim': list[str],    # taam names on this word
        'joined': bool }        # True if connected to next word by maqaf

    Munah-legarmeh detection (requires verse-level context):
      A munah+paseq word is munah-legarmeh only when the next taam-bearing
      word carries revii or munah→revii (i.e. the legarmeh governs toward revii).
      Otherwise munah and paseq are kept as separate marks (emphatic paseq).
    """
    raw_words = verse_text.split()
    words = []
    for raw in raw_words:
        # Skip section markers {ס}/{פ} and ketiv readings (word)
        if raw.startswith('{') or raw.startswith('('):
            continue
        # Strip keri brackets [word] → word (keep the taam-bearing reading)
        if raw.startswith('[') and ']' in raw:
            raw = raw.replace('[', '').replace(']', '')
        plain = strip_nikud_and_teamim(raw).replace(MAQAF, '')
        plain = plain.replace('\u05C3', '').replace('\u05C0', '').strip()
        if not plain:
            continue
        words.append({
            'text':   raw,
            'plain':  plain,
            'teamim': word_teamim(raw),
            'joined': raw.endswith(MAQAF),
        })

    # Promote munah+paseq → munah-legarmeh when followed by revii (possibly
    # with an intervening munah), i.e. the pattern: [munah+paseq] ... revii
    for i, w in enumerate(words):
        t = w['teamim']
        if 'munah' in t and 'paseq' in t:
            # Look ahead: skip words with no teamim, check if next taam-bearing
            # word is revii, or munah followed eventually by revii
            lookahead = [ww['teamim'] for ww in words[i+1:] if ww['teamim']]
            governed_by_revii = False
            for j, future_taams in enumerate(lookahead):
                if 'revii' in future_taams:
                    governed_by_revii = True
                    break
                if any(tt not in ('munah', 'paseq') for tt in future_taams):
                    # Hit a non-conjunctive taam before revii — not legarmeh
                    break
            if governed_by_revii:
                w['teamim'] = [tt for tt in t if tt not in ('munah', 'paseq')]
                w['teamim'].append('munah-legarmeh')

    return words


def verse_taam_sequence(verse_text: str) -> list[str]:
    """Return the ordered sequence of taam names across the whole verse."""
    seq = []
    for w in verse_word_list(verse_text):
        seq.extend(w['teamim'])
    return seq


# ---------------------------------------------------------------------------
# Corpus iteration helpers
# ---------------------------------------------------------------------------

def iter_verses(books=None, corpus=None):
    """
    Yield (book, chapter, verse_num, text) for every verse in the corpus.
    books: list of book codes to include; None = all
    corpus: pre-loaded dict; None = load_tikkun_data()
    """
    if corpus is None:
        corpus = load_tikkun_data()
    for book, chapters in corpus.items():
        if books and book not in books:
            continue
        for chap, verses in chapters.items():
            for vnum, text in verses.items():
                yield book, chap, vnum, text


def build_verse_index(books=None) -> list[dict]:
    """
    Build a flat list of verse records, each a dict with keys:
      book, chapter, verse, text, words, taam_sequence
    """
    records = []
    for book, chap, vnum, text in iter_verses(books=books):
        words = verse_word_list(text)
        taam_seq = [t for w in words for t in w['teamim']]
        records.append({
            'book':          book,
            'chapter':       int(chap),
            'verse':         int(vnum),
            'text':          text,
            'words':         words,
            'taam_sequence': taam_seq,
            'n_words':       len(words),
        })
    return records
