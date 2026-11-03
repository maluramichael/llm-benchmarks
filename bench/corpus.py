"""Pinned English Wikipedia corpus: download, verify, cache, cut to token length.

The corpus is long filler text for the context ladder and needle tests. The
text itself is CC BY-SA and is never committed. Only `bench/data/corpus.toml`
is: article title, revision id and the sha256 of the plain text.

Determinism: every machine must get byte-identical text. The API call asks for
the wikitext of one fixed revision (`action=query&revids=N&prop=revisions`),
which never changes. The TextExtracts API (`prop=extracts`) ignores the
revision and always renders the current page, and `action=parse&oldid=N`
renders old wikitext with today's templates, so neither is stable. The
wikitext is turned into plain text by `wikitext_to_text`, a deterministic
stdlib converter; its output is pinned by the sha256 in the manifest. If the
converter changes, the checksums must be pinned again (`python -m bench.corpus
pin`).

Cache layout: `<cache_dir>/<revid>.txt`, one file per article, UTF-8, Unix
newlines, content exactly the plain text. A directory copied from a mirror
works as is, because every file is verified against the manifest.
"""

from __future__ import annotations

import hashlib
import html
import http.client
import json
import os
import re
import ssl
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from bench.store import retry_on_lock

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "corpus.toml"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "llm-benchmarks/1.0 (https://github.com/maluramichael/llm-benchmarks)"
REQUEST_PAUSE = 0.5  # seconds between requests, sequential and polite
ATTEMPTS = 5
BACKOFF_START = 2.0
BACKOFF_MAX = 60.0
TIMEOUT = 60
CHARS_PER_TOKEN_GUESS = 4  # first guess for English text
SEARCH_STEPS = 30  # hard stop for the cut search
SENTINEL = "\x00"  # marks removed markup, so lines left empty can be dropped


class CorpusError(RuntimeError):
    """The corpus cannot be loaded (missing offline, bad manifest, API trouble)."""


@dataclass(frozen=True)
class Article:
    title: str
    revid: int
    sha256: str


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def read_manifest(path: Path = DEFAULT_MANIFEST) -> list[Article]:
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise CorpusError(f"cannot read corpus manifest {path}: {err}") from err
    entries = data.get("article")
    if not isinstance(entries, list) or not entries:
        raise CorpusError(f"corpus manifest {path} has no [[article]] entries")
    articles: list[Article] = []
    for entry in entries:
        title, revid, digest = entry.get("title"), entry.get("revid"), entry.get("sha256")
        if not isinstance(title, str) or not title:
            raise CorpusError(f"corpus manifest {path}: article without title: {entry}")
        if not isinstance(revid, int) or isinstance(revid, bool) or revid <= 0:
            raise CorpusError(f"corpus manifest {path}: {title}: revid must be a positive integer")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CorpusError(f"corpus manifest {path}: {title}: sha256 must be 64 lowercase hex digits")
        articles.append(Article(title, revid, digest))
    for field in ("title", "revid"):
        values = [getattr(a, field) for a in articles]
        if len(values) != len(set(values)):
            raise CorpusError(f"corpus manifest {path}: duplicate {field}")
    return articles


def render_manifest(articles: Sequence[Article]) -> str:
    lines = [
        "# Pinned English Wikipedia corpus for the long-context tests.",
        "#",
        "# Source: English Wikipedia, https://en.wikipedia.org/w/index.php?oldid=<revid>",
        "# License of the article text: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/),",
        "# authors are listed in each article history. The text is NOT part of this repository:",
        "# it is downloaded at run time and cached under .bench/cache/corpus/<revid>.txt.",
        "# This file only holds title, revision id and the sha256 of the plain text, so every",
        "# machine verifies that it works on byte-identical text.",
        "#",
        "# sha256 covers the UTF-8 output of bench.corpus.wikitext_to_text for that revision.",
        "# To pin other articles: python -m bench.corpus pin <title> [<title> ...]",
        "",
    ]
    for article in articles:
        lines += ["[[article]]", f"title = {json.dumps(article.title)}",
                  f"revid = {article.revid}", f'sha256 = "{article.sha256}"', ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Wikitext to plain text
# --------------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_BLOCK_TAGS = re.compile(
    r"<(ref|gallery|math|timeline|imagemap|score|syntaxhighlight|source|pre|chem|ce|hiero"
    r"|templatedata|mapframe|maplink|graph)\b[^>]*?(?:/>|>.*?</\1\s*>)",
    re.DOTALL | re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_EXTERNAL_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:[ \t]+([^\]\n]*))?\]")
_QUOTES = re.compile(r"'{2,}")
_MAGIC_WORD = re.compile(r"__[A-Z]+__")
_HEADING = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$")
_LIST_MARKER = re.compile(r"^[*#:;]+\s*")
_RULE = re.compile(r"^-{4,}\s*$")
_BLANKS = re.compile(r"\n{3,}")
_DROPPED_SECTIONS = {
    "see also", "references", "notes", "footnotes", "external links", "further reading",
    "sources", "bibliography", "citations", "literature", "gallery", "notes and references",
    "references and notes", "works cited", "general bibliography",
}
_UNIT_RANGE = {"to", "and", "or", "-", "–", "x", "by", "+", "±"}
_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", "​": None,
                         "﻿": None})


def _find_close(text: str, start: int, opener: str, closer: str, token: re.Pattern) -> int:
    """Index of the closer matching the opener at `start`, or -1."""
    depth = 0
    for match in token.finditer(text, start):
        depth += 1 if match.group() == opener else -1
        if depth == 0:
            return match.start()
    return -1


def _replace_nested(text: str, opener: str, closer: str,
                    handle: Callable[[str], str]) -> str:
    """Replace every outermost opener...closer pair by handle(inner).

    Inner pairs are handled first. An opener without closer is dropped.
    """
    token = re.compile(re.escape(opener) + "|" + re.escape(closer))
    out: list[str] = []
    i = 0
    while True:
        start = text.find(opener, i)
        if start < 0:
            out.append(text[i:])
            return "".join(out)
        end = _find_close(text, start, opener, closer, token)
        out.append(text[i:start])
        if end < 0:
            i = start + len(opener)
            continue
        inner = _replace_nested(text[start + len(opener):end], opener, closer, handle)
        out.append(handle(inner))
        i = end + len(closer)


def _split_args(inner: str) -> list[str]:
    """Split on top level pipes, keeping [[a|b]] links together."""
    parts, depth, current, i = [], 0, [], 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two == "[[":
            depth += 1
            current.append(two)
            i += 2
        elif two == "]]" and depth:
            depth -= 1
            current.append(two)
            i += 2
        elif inner[i] == "|" and depth == 0:
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(inner[i])
            i += 1
    parts.append("".join(current))
    return parts


def _template(inner: str) -> str:
    """Keep the visible text of a few templates; drop all others."""
    args = _split_args(inner)
    name = re.sub(r"[\s_]+", " ", args[0]).strip().lower()
    named = {}
    positional = []
    for arg in args[1:]:
        key, eq, value = arg.partition("=")
        if eq and re.fullmatch(r"\s*[A-Za-z][\w ]*\s*", key):
            named[key.strip().lower()] = value.strip()
        else:
            positional.append(arg.strip())
    if name in ("convert", "cvt"):
        if len(positional) >= 4 and positional[1] in _UNIT_RANGE:
            return " ".join(positional[:4])
        if len(positional) >= 2:
            return f"{positional[0]} {positional[1]}"
    elif name in ("lang", "langx", "transliteration", "transl") and len(positional) >= 2:
        return positional[1]
    elif name.startswith("lang-") and positional:
        return positional[0]
    elif name in ("nowrap", "nobr", "small", "abbr", "smallcaps", "em", "strong") and positional:
        return positional[0]
    elif name in ("circa", "c.") and positional:
        return "c. " + positional[0]
    elif name in ("nbsp",):
        return " "
    elif name in ("snd", "spaced ndash", "spnd"):
        return " – "
    elif name in ("quote", "blockquote", "cquote", "quotation", "quote box"):
        return named.get("text") or (positional[0] if positional else "")
    return SENTINEL


def _link(inner: str) -> str:
    target, _, rest = inner.partition("|")
    bare = target.strip().lstrip(":").strip()
    if not target.strip().startswith(":"):
        namespace = bare.split(":", 1)[0].lower() if ":" in bare else ""
        if namespace in ("file", "image", "media", "category"):
            return SENTINEL
    label = rest.rsplit("|", 1)[-1].strip() if "|" in rest or rest else ""
    return label or bare


def _remove_tables(text: str) -> str:
    out, depth = [], 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("{|"):
            if depth == 0:
                out.append(SENTINEL)
            depth += 1
        elif depth and stripped.startswith("|}"):
            depth -= 1
        elif depth == 0:
            out.append(line)
    return "\n".join(out)


def _squeeze(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def wikitext_to_text(wikitext: str) -> str:
    """Deterministic, stdlib only conversion of wikitext to readable plain text.

    Drops templates (except a few that carry visible text), references,
    tables, files, categories, and the sections "See also", "References",
    "External links" and similar. Keeps headings (as plain lines), paragraphs
    and list items (without markers).
    """
    text = wikitext.replace("\r\n", "\n").replace("\r", "\n").replace(SENTINEL, "")
    text = _COMMENT.sub(SENTINEL, text)
    text = _BLOCK_TAGS.sub(SENTINEL, text)
    text = _replace_nested(text, "{{", "}}", _template)
    text = _remove_tables(text)
    text = _replace_nested(text, "[[", "]]", _link)
    text = _EXTERNAL_LINK.sub(lambda m: (m.group(1) or "").strip(), text)

    lines = [line for line in text.split("\n")
             if not (SENTINEL in line and not line.replace(SENTINEL, "").strip())]
    text = "\n".join(lines).replace(SENTINEL, "")
    text = _MAGIC_WORD.sub("", text)

    out: list[str] = []
    skip_level = 0
    for line in text.split("\n"):
        heading = _HEADING.match(line.strip())
        if heading:
            level, title = len(heading.group(1)), _squeeze(_QUOTES.sub("", heading.group(2)))
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and title.lower() in _DROPPED_SECTIONS:
                skip_level = level
            if not skip_level:
                out.append(title)
            continue
        if skip_level or _RULE.match(line.strip()):
            continue
        out.append(_squeeze(_LIST_MARKER.sub("", line.lstrip())))
    text = "\n".join(out)

    text = _QUOTES.sub("", text)
    text = _BR.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text).translate(_SPACES)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\(\s*[;,]\s*", "(", text)
    text = re.sub(r"\s*[;,]\s*\)", ")", text)
    text = re.sub(r"[ ]+([,;])", r"\1", text)
    text = re.sub(r" +\.(?=\s|$)", ".", text)
    for marker in ("{{", "}}", "[[", "]]"):
        text = text.replace(marker, "")
    text = "\n".join(_squeeze(line) for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


# --------------------------------------------------------------------------
# MediaWiki API
# --------------------------------------------------------------------------

def _api_get(params: dict, what: str, pause: float) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(params, safe="|")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = BACKOFF_START
    last = "no attempt"
    for attempt in range(ATTEMPTS):
        if pause > 0:
            time.sleep(pause)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            last = f"HTTP {err.code}"
            if err.code != 429 and err.code < 500:
                raise CorpusError(f"{what}: {last}") from err
            retry_after = (getattr(err, "headers", None) or {}).get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
            delay = min(delay * 2, BACKOFF_MAX)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError,
                http.client.HTTPException) as err:
            # also a read cut short (IncompleteRead) or a broken TLS stream: retried
            last = str(getattr(err, "reason", None) or err or type(err).__name__)
            wait = delay
            delay = min(delay * 2, BACKOFF_MAX)
        except (ValueError, UnicodeDecodeError) as err:
            raise CorpusError(f"{what}: unreadable API response: {err}") from err
        else:
            if "error" in data:
                raise CorpusError(f"{what}: API error: {data['error'].get('info', data['error'])}")
            return data
        if attempt + 1 < ATTEMPTS:
            time.sleep(min(wait, BACKOFF_MAX))
    raise CorpusError(f"{what}: giving up after {ATTEMPTS} attempts ({last})")


def fetch_wikitext(title: str, revid: int, pause: float = REQUEST_PAUSE) -> str:
    """Wikitext of exactly this revision of the article."""
    what = f"corpus article {title!r} (revision {revid})"
    data = _api_get({"action": "query", "revids": revid, "prop": "revisions",
                     "rvprop": "ids|content", "rvslots": "main", "format": "json",
                     "formatversion": 2}, what, pause)
    query = data.get("query", {})
    pages = query.get("pages") or []
    if query.get("badrevids") or not pages:
        raise CorpusError(f"{what}: revision does not exist (any more)")
    page = pages[0]
    if page.get("title") != title:
        # Not an error: revid and the sha256 of the text pin the content, and a page move
        # only changes the title.
        print(f"note: {what}: the page is now called {page.get('title')!r}", file=sys.stderr)
    try:
        revision = page["revisions"][0]
        content = revision["slots"]["main"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise CorpusError(f"{what}: revision has no readable content") from err
    if revision.get("revid") != revid:
        raise CorpusError(f"{what}: API returned revision {revision.get('revid')}")
    return content


def latest_revid(title: str, pause: float = REQUEST_PAUSE) -> int:
    what = f"corpus article {title!r}"
    data = _api_get({"action": "query", "titles": title, "prop": "revisions",
                     "rvprop": "ids", "format": "json", "formatversion": 2}, what, pause)
    pages = data.get("query", {}).get("pages") or []
    if not pages or pages[0].get("missing") or pages[0].get("title") != title:
        raise CorpusError(f"{what}: no such page (redirect or typo?)")
    return pages[0]["revisions"][0]["revid"]


def pin_article(title: str, latest: Callable[[str], int] = latest_revid,
                fetch: Callable[[str, int], str] = fetch_wikitext) -> Article:
    """Pin the current revision of a page: revision id and checksum."""
    revid = latest(title)
    wikitext = fetch(title, revid)
    if wikitext.lstrip().upper().startswith("#REDIRECT"):
        raise CorpusError(f"corpus article {title!r} is a redirect; pin the target page")
    return Article(title, revid, _digest(wikitext_to_text(wikitext)))


# --------------------------------------------------------------------------
# Loading and caching
# --------------------------------------------------------------------------

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checked(article: Article, text: str, origin: str, cached: bool = False) -> str:
    actual = _digest(text)
    if actual != article.sha256:
        hint = (f"; delete the cache file {article.revid}.txt and run again" if cached else "")
        raise ValueError(
            f"corpus article {article.title!r} (revision {article.revid}): sha256 mismatch in "
            f"{origin}: expected {article.sha256}, got {actual}{hint}")
    return text


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    def write() -> None:
        with open(tmp, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    retry_on_lock(write)  # a virus scanner may hold the file for a moment on Windows
    retry_on_lock(lambda: os.replace(tmp, path))


def load_corpus(cache_dir: Path, offline: bool = False, *, manifest: Path | None = None,
                fetch: Callable[[str, int], str] | None = None,
                progress: Callable[[str], None] | None = None) -> str:
    """All pinned articles as one text, in manifest order.

    Missing articles are downloaded (sequentially) into `cache_dir`, unless
    `offline` is set; then an incomplete cache raises `CorpusError` naming the
    missing articles. A file whose sha256 differs from the manifest raises
    `ValueError` naming the article; nothing is deleted or overwritten.
    """
    articles = read_manifest(manifest or DEFAULT_MANIFEST)
    cache_dir = Path(cache_dir)
    bodies: dict[int, str] = {}
    missing: list[Article] = []
    for article in articles:
        path = cache_dir / f"{article.revid}.txt"
        if not path.exists():
            missing.append(article)
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as err:
            raise ValueError(f"corpus article {article.title!r} (revision {article.revid}): "
                             f"{path} is not valid UTF-8") from err
        bodies[article.revid] = _checked(article, text, str(path), cached=True)
    if missing and offline:
        names = ", ".join(f"{a.title} ({a.revid}.txt)" for a in missing)
        raise CorpusError(f"corpus cache {cache_dir} is incomplete and offline mode is on; "
                          f"missing {len(missing)} of {len(articles)}: {names}")
    fetch = fetch or fetch_wikitext
    for number, article in enumerate(missing, 1):
        if progress:
            progress(f"downloading corpus article {article.title} ({number}/{len(missing)})")
        text = _checked(article, wikitext_to_text(fetch(article.title, article.revid)),
                        "the download")
        _write_atomic(cache_dir / f"{article.revid}.txt", text)
        bodies[article.revid] = text
    return "\n\n".join(f"{a.title}\n\n{bodies[a.revid]}" for a in articles)


# --------------------------------------------------------------------------
# Cutting to a token length
# --------------------------------------------------------------------------

def _snap(text: str, k: int) -> int:
    """Largest cut position <= k that sits on whitespace without trailing blanks."""
    if k >= len(text):
        return len(text)
    while k > 0 and not text[k].isspace():
        k -= 1
    while k > 0 and text[k - 1].isspace():
        k -= 1
    return k


def cut_to_tokens(text: str, n_tokens: int, count: Callable[[str], int]) -> str:
    """Longest prefix of `text` (cut at whitespace) with count(prefix) <= n_tokens.

    `count` asks the model's tokenizer and is expensive, so the search is
    guided: the first guess is 4 characters per token, every later guess is
    interpolated from what the counter already said, inside a bracket of
    "fits" and "does not fit" positions. It stops as soon as the result is
    within 0.5 % of the budget. Typical cost is 2 to 4 calls. The returned
    prefix was always measured, never assumed. A text that already fits is
    returned unchanged.
    """
    if n_tokens < 0:
        raise ValueError("n_tokens must not be negative")
    if n_tokens == 0 or not text:
        return ""
    tolerance = max(1, n_tokens // 200)
    target = n_tokens - tolerance // 2  # aim inside the accepted window
    fit_pos, fit_count = 0, 0
    over_pos, over_count = None, None
    guess = n_tokens * CHARS_PER_TOKEN_GUESS
    for step in range(SEARCH_STEPS):
        pos = _snap(text, guess)
        if pos <= fit_pos or (over_pos is not None and pos >= over_pos):
            break  # no new whitespace position between the bracket ends
        used = count(text[:pos])
        if used <= n_tokens:
            fit_pos, fit_count = pos, used
            if used >= n_tokens - tolerance or pos == len(text):
                break
        else:
            over_pos, over_count = pos, used
        if step >= 8 and over_pos is not None:
            guess = (fit_pos + over_pos) // 2  # plain bisection as a safeguard
        elif over_pos is None:
            # a counter that says 0 gives no ratio: just double the prefix
            guess = max(fit_pos + 1, int(fit_pos * target / fit_count) if fit_count
                        else fit_pos * 2)
        elif fit_pos == 0:
            guess = int(over_pos * target / over_count)
        else:
            slope = (over_pos - fit_pos) / (over_count - fit_count)
            guess = int(fit_pos + (target - fit_count) * slope)
        if over_pos is not None:
            guess = min(max(guess, fit_pos + 1), over_pos - 1)
    return text if fit_pos == len(text) else text[:fit_pos]


def needle_text(text: str, n_tokens: int, depth: float, needle: str,
                count: Callable[[str], int]) -> str:
    """A prompt of at most `n_tokens` tokens: corpus prefix with `needle` inserted.

    `depth` is the relative position of the needle (0.0 start, 1.0 end). The
    needle goes into its own paragraph, at the paragraph break nearest to the
    requested depth. The total is measured, and the corpus part is shortened
    until the whole text fits.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between 0.0 and 1.0")
    budget = n_tokens - count("\n\n" + needle + "\n\n")
    if budget <= 0:
        raise ValueError(f"needle does not fit into {n_tokens} tokens")
    for _ in range(6):
        body = cut_to_tokens(text, budget, count)
        result = _insert(body, needle, depth)
        total = count(result)
        if total <= n_tokens:
            return result
        budget -= total - n_tokens + 1
        if budget <= 0:
            break
    raise ValueError(f"cannot fit corpus text and needle into {n_tokens} tokens")


def _insert(body: str, needle: str, depth: float) -> str:
    if depth <= 0.0 or not body:
        return needle + "\n\n" + body
    if depth >= 1.0:
        return body + "\n\n" + needle
    target = int(len(body) * depth)
    before = body.rfind("\n\n", 0, target)
    after = body.find("\n\n", target)
    candidates = [p for p in (before, after) if p >= 0]
    if not candidates:  # no paragraph breaks at all: fall back to whitespace
        pos = _snap(body, target)
        return body[:pos] + "\n\n" + needle + "\n\n" + body[pos:].lstrip()
    pos = min(candidates, key=lambda p: abs(p - target))
    return body[:pos] + "\n\n" + needle + body[pos:]


# --------------------------------------------------------------------------
# Maintenance command: python -m bench.corpus pin <title> ...
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "pin":
        print("usage: python -m bench.corpus pin <title> [<title> ...]\n"
              "Pins the current revision of each article and prints a corpus.toml.",
              file=sys.stderr)
        return 2
    pinned = []
    for title in args[1:]:
        article = pin_article(title)
        print(f"pinned {article.title} at revision {article.revid}", file=sys.stderr)
        pinned.append(article)
    sys.stdout.write(render_manifest(pinned))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
