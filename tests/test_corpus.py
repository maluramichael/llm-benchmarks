import hashlib
import io
import json
import os
import tempfile
import tomllib
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from bench import corpus
from bench.corpus import (CorpusError, cut_to_tokens, load_corpus, needle_text,
                          wikitext_to_text)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "bench" / "data" / "corpus.toml"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chars4(text: str) -> int:
    """Fake tokenizer: one token per 4 characters, rounded up."""
    return -(-len(text) // 4)


class Counter:
    """Wraps a count function and remembers how often it was asked."""

    def __init__(self, fn=chars4):
        self.fn = fn
        self.calls = 0

    def __call__(self, text: str) -> int:
        self.calls += 1
        return self.fn(text)


def prose(n_words: int) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima"]
    out = []
    for i in range(n_words):
        out.append(words[i % len(words)] + str(i % 7))
        if i % 40 == 39:
            out.append("\n\n")
    return " ".join(out)


def mixed_tokens(text: str) -> int:
    """Uneven tokenizer: words cost 1 token, digits and punctuation cost more."""
    return sum(1 + len(w) // 6 for w in text.split()) + text.count("\n\n")


class CutTest(unittest.TestCase):
    def setUp(self):
        self.text = prose(20000)

    def test_cut_exact(self):
        count = Counter()
        out = cut_to_tokens(self.text, 100, count)
        self.assertLessEqual(chars4(out), 100)
        self.assertGreater(chars4(out), 95)
        self.assertTrue(self.text.startswith(out))

    def test_cut_on_whitespace(self):
        for n in (10, 100, 777, 5000):
            with self.subTest(n=n):
                out = cut_to_tokens(self.text, n, chars4)
                self.assertTrue(self.text.startswith(out))
                self.assertTrue(self.text[len(out)].isspace())
                self.assertEqual(out, out.rstrip())

    def test_short_text_is_returned_whole(self):
        count = Counter()
        self.assertEqual(cut_to_tokens("one two three", 100, count), "one two three")
        self.assertEqual(count.calls, 1)

    def test_counter_that_returns_zero_does_not_divide_by_zero(self):
        out = cut_to_tokens(self.text, 100, lambda t: 0)
        self.assertTrue(self.text.startswith(out))
        self.assertGreater(len(out), 0)

    def test_zero_budget(self):
        self.assertEqual(cut_to_tokens(self.text, 0, chars4), "")

    def test_call_count_is_small(self):
        # Every call is a tokenizer round trip through the engine, so the
        # search must converge in a handful of calls even for a 256k cut.
        text = prose(400000)
        for n, limit in ((100, 6), (8192, 6), (256000, 6)):
            with self.subTest(n=n):
                count = Counter()
                out = cut_to_tokens(text, n, count)
                self.assertLessEqual(chars4(out), n)
                self.assertGreater(chars4(out), n * 0.995 - 1)
                self.assertLessEqual(count.calls, limit)

    def test_call_count_when_the_first_guess_is_far_off(self):
        # 2.5 and 7 characters per token: the first guess (4) is wrong in
        # both directions, interpolation must still land in 4 calls or fewer.
        text = prose(400000)
        for chars_per_token in (2.5, 7.0):
            def tokens(t, ratio=chars_per_token):
                return int(len(t) / ratio) + 1
            for n in (8192, 256000):
                with self.subTest(ratio=chars_per_token, n=n):
                    count = Counter(tokens)
                    out = cut_to_tokens(text, n, count)
                    self.assertLessEqual(tokens(out), n)
                    self.assertGreater(tokens(out), n * 0.995 - 1)
                    self.assertLessEqual(count.calls, 4)

    def test_call_count_with_uneven_tokenizer(self):
        text = prose(400000)
        count = Counter(mixed_tokens)
        out = cut_to_tokens(text, 50000, count)
        self.assertLessEqual(mixed_tokens(out), 50000)
        self.assertGreater(mixed_tokens(out), 50000 * 0.99)
        self.assertLessEqual(count.calls, 10)

    def test_result_is_always_measured_to_fit(self):
        # A tokenizer that is not linear in the prefix length must still
        # never produce a result above the budget.
        def odd(text: str) -> int:
            return len(text) // 4 + (len(text) % 7)
        for n in (50, 999, 12345):
            with self.subTest(n=n):
                out = cut_to_tokens(self.text, n, odd)
                self.assertLessEqual(odd(out), n)


class NeedleTest(unittest.TestCase):
    NEEDLE = "The secret passphrase is MAGENTA-TORTOISE-42."

    def setUp(self):
        self.text = prose(40000)

    def test_needle_depth(self):
        out = needle_text(self.text, 2000, 0.5, self.NEEDLE, chars4)
        pos = out.index(self.NEEDLE)
        self.assertGreater(pos, len(out) * 0.25)
        self.assertLess(pos, len(out) * 0.75)

    def test_needle_depth_close_to_target(self):
        for depth in (0.0, 0.1, 0.5, 0.9, 1.0):
            with self.subTest(depth=depth):
                out = needle_text(self.text, 4000, depth, self.NEEDLE, chars4)
                got = out.index(self.NEEDLE) / len(out)
                self.assertAlmostEqual(got, depth, delta=0.05)

    def test_total_fits_budget_and_is_close(self):
        for n in (500, 4000, 30000):
            with self.subTest(n=n):
                count = Counter()
                out = needle_text(self.text, n, 0.3, self.NEEDLE, count)
                self.assertLessEqual(chars4(out), n)
                self.assertGreater(chars4(out), n * 0.97)
                self.assertEqual(out.count(self.NEEDLE), 1)
                self.assertLessEqual(count.calls, 12)

    def test_needle_sits_in_its_own_paragraph(self):
        out = needle_text(self.text, 2000, 0.5, self.NEEDLE, chars4)
        self.assertIn("\n\n" + self.NEEDLE + "\n\n", out)

    def test_bad_depth(self):
        for depth in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                needle_text(self.text, 2000, depth, self.NEEDLE, chars4)

    def test_needle_larger_than_budget(self):
        with self.assertRaises(ValueError):
            needle_text(self.text, 5, 0.5, self.NEEDLE, chars4)

    def test_body_is_a_prefix_of_the_corpus(self):
        out = needle_text(self.text, 2000, 0.5, self.NEEDLE, chars4)
        stripped = out.replace("\n\n" + self.NEEDLE + "\n\n", "\n\n")
        head = stripped[:200]
        self.assertTrue(self.text.startswith(head))


class WikitextTest(unittest.TestCase):
    def test_templates_refs_comments_removed(self):
        raw = ("{{short description|River}}\n{{Infobox river\n| name = X\n"
               "| nested = {{a|{{b}}}}\n}}\nThe river<ref name=a>Cite {{cite web|url=x}}</ref> "
               "flows.<ref name=b />\n<!-- hidden -->Next.")
        self.assertEqual(wikitext_to_text(raw), "The river flows.\nNext.")

    def test_kept_templates(self):
        raw = ("It is {{convert|1,230|km|mi|abbr=on}} long, {{lang|de|Rhein}}, "
               "{{nowrap|a b}}, {{circa|1200}}, {{cvt|5|m}}.")
        self.assertEqual(wikitext_to_text(raw),
                         "It is 1,230 km long, Rhein, a b, c. 1200, 5 m.")

    def test_links(self):
        raw = ("[[Rhine]] and [[Lake Constance|the lake]] and [[Category:Rivers]]\n"
               "[[File:Map.png|thumb|A map of [[Germany]] here]]\n"
               "[[Image:Old.jpg|thumb|caption]]After. [[:Category:Foo|Foo]] [[a|]]")
        out = wikitext_to_text(raw)
        self.assertEqual(out, "Rhine and the lake and\nAfter. Foo a")

    def test_external_links_and_markup(self):
        raw = "See [https://example.org the site] or [https://example.org] and '''bold''' ''it'' '''''both'''''."
        self.assertEqual(wikitext_to_text(raw), "See the site or and bold it both.")

    def test_headings_and_dropped_sections(self):
        raw = ("Lead text.\n\n== History ==\nOld times.\n\n=== Early ===\nEarly.\n\n"
               "== See also ==\n* [[Other]]\n\n== References ==\n{{reflist}}\n\n"
               "== Geography ==\nFlat.\n\n== External links ==\n* [http://x y]\n")
        out = wikitext_to_text(raw)
        self.assertEqual(out, "Lead text.\n\nHistory\nOld times.\n\nEarly\nEarly.\n\nGeography\nFlat.")

    def test_dropped_section_keeps_deeper_subsections_dropped(self):
        raw = "Intro.\n\n== Notes ==\nn\n=== Sub ===\ns\n== After ==\nText."
        self.assertEqual(wikitext_to_text(raw), "Intro.\n\nAfter\nText.")

    def test_lists(self):
        raw = "* one\n** two\n# three\n; term\n: definition\nplain"
        self.assertEqual(wikitext_to_text(raw), "one\ntwo\nthree\nterm\ndefinition\nplain")

    def test_tables_removed(self):
        raw = "Before.\n{| class=\"wikitable\"\n|-\n| a || b\n{|\n| nested\n|}\n|}\nAfter."
        self.assertEqual(wikitext_to_text(raw), "Before.\nAfter.")

    def test_html_and_entities(self):
        raw = ("H<sub>2</sub>O&nbsp;is &amp; wet.<br />Next<br>line "
               "<gallery>\nFile:a.jpg|x\n</gallery>"
               "<math>x^2</math> <div class=\"x\">div text</div>")
        out = wikitext_to_text(raw)
        self.assertIn("H2O is & wet.", out)
        self.assertNotIn("<", out)
        self.assertNotIn("File:a.jpg", out)
        self.assertIn("div text", out)

    def test_empty_parentheses_and_spaces_cleaned(self):
        raw = "The Rhine ({{lang|de|Rhein}}; {{IPAc-en|x}}) is long ( ) and  wide ,yes."
        out = wikitext_to_text(raw)
        self.assertNotIn("()", out)
        self.assertNotIn("  ", out)
        self.assertIn("The Rhine (Rhein)", out)

    def test_deterministic_and_normalized(self):
        raw = "A\r\nB   C\t\n\n\n\n\nD  \n"
        out = wikitext_to_text(raw)
        self.assertEqual(out, "A\nB C\n\nD")
        self.assertEqual(out, wikitext_to_text(raw))

    def test_no_markup_left(self):
        raw = "{{a}}[[b]]<ref>c</ref>''d'' {{unclosed template and [[unclosed link"
        out = wikitext_to_text(raw)
        for marker in ("{{", "}}", "[[", "]]", "<ref", "''"):
            self.assertNotIn(marker, out)


class ApiTest(unittest.TestCase):
    def fake_urlopen(self, payload, seen):
        def urlopen(request, timeout=None):
            seen.append(request)
            return io.BytesIO(json.dumps(payload).encode("utf-8"))
        return urlopen

    def payload(self, title="Rhine", revid=7, content="Text"):
        return {"batchcomplete": True, "query": {"pages": [{
            "pageid": 1, "title": title,
            "revisions": [{"revid": revid, "slots": {"main": {"content": content}}}]}]}}

    def test_request_pins_the_revision(self):
        seen = []
        with mock.patch("urllib.request.urlopen", self.fake_urlopen(self.payload(), seen)):
            text = corpus.fetch_wikitext("Rhine", 7, pause=0)
        self.assertEqual(text, "Text")
        request = seen[0]
        self.assertIn("revids=7", request.full_url)
        self.assertIn("rvslots=main", request.full_url)
        self.assertNotIn("titles=", request.full_url)
        self.assertTrue(request.full_url.startswith("https://en.wikipedia.org/w/api.php?"))
        self.assertIn("llm-benchmarks/1.0", request.get_header("User-agent"))

    def test_moved_page_is_only_a_note(self):
        # revid and sha256 pin the content; after a page move the title differs
        err = io.StringIO()
        with mock.patch("urllib.request.urlopen",
                        self.fake_urlopen(self.payload(title="Danube"), [])), \
                mock.patch("sys.stderr", err):
            self.assertEqual(corpus.fetch_wikitext("Rhine", 7, pause=0), "Text")
        self.assertIn("Danube", err.getvalue())
        self.assertIn("Rhine", err.getvalue())

    def test_broken_reads_are_retried_and_then_wrapped(self):
        import http.client
        import ssl
        good = self.fake_urlopen(self.payload(), [])
        for exc in (http.client.IncompleteRead(b"ab"), ssl.SSLError("bad record mac"),
                    ConnectionResetError("reset")):
            calls = []

            def flaky(request, timeout=None, exc=exc):
                calls.append(1)
                if len(calls) == 1:
                    raise exc
                return good(request)

            with mock.patch("urllib.request.urlopen", flaky), mock.patch("time.sleep"):
                self.assertEqual(corpus.fetch_wikitext("Rhine", 7, pause=0), "Text")
            self.assertEqual(len(calls), 2, exc)

            def broken(request, timeout=None, exc=exc):
                raise exc

            with mock.patch("urllib.request.urlopen", broken), mock.patch("time.sleep"):
                with self.assertRaisesRegex(CorpusError, "Rhine"):
                    corpus.fetch_wikitext("Rhine", 7, pause=0)

    def test_missing_revision_is_an_error(self):
        payload = {"batchcomplete": True, "query": {"badrevids": {"7": {"revid": 7}}}}
        with mock.patch("urllib.request.urlopen", self.fake_urlopen(payload, [])):
            with self.assertRaisesRegex(CorpusError, "Rhine"):
                corpus.fetch_wikitext("Rhine", 7, pause=0)

    def test_retries_on_429_then_succeeds(self):
        calls = []
        good = self.fake_urlopen(self.payload(), [])

        def urlopen(request, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError(request.full_url, 429, "slow down",
                                             {"Retry-After": "0"}, None)
            return good(request)

        with mock.patch("urllib.request.urlopen", urlopen), \
                mock.patch("time.sleep") as sleep:
            self.assertEqual(corpus.fetch_wikitext("Rhine", 7, pause=0), "Text")
        self.assertEqual(len(calls), 3)
        self.assertTrue(sleep.called)

    def test_gives_up_after_attempts(self):
        def urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)

        with mock.patch("urllib.request.urlopen", urlopen), mock.patch("time.sleep"):
            with self.assertRaisesRegex(CorpusError, "Rhine"):
                corpus.fetch_wikitext("Rhine", 7, pause=0)

    def test_client_error_is_not_retried(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 404, "gone", {}, None)

        with mock.patch("urllib.request.urlopen", urlopen), mock.patch("time.sleep"):
            with self.assertRaises(CorpusError):
                corpus.fetch_wikitext("Rhine", 7, pause=0)
        self.assertEqual(len(calls), 1)


ARTICLES = {
    "Alpha": ("{{x}}Alpha is '''first'''.\n\n== Part ==\nMore alpha.", 11),
    "Beta River": ("Beta is [[Beta|second]].", 22),
    "Gamma": ("Gamma &amp; co.", 33),
}


class CorpusTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.cache = self.dir / "cache" / "corpus"
        self.fetched = []
        self.plain = {t: wikitext_to_text(w) for t, (w, _) in ARTICLES.items()}
        self.manifest = self.dir / "corpus.toml"
        self.write_manifest()

    def write_manifest(self, overrides=None):
        overrides = overrides or {}
        lines = []
        for title, (_, revid) in ARTICLES.items():
            digest = overrides.get(title, sha(self.plain[title]))
            lines += ["[[article]]", f'title = "{title}"', f"revid = {revid}",
                      f'sha256 = "{digest}"', ""]
        self.manifest.write_text("\n".join(lines), encoding="utf-8")

    def fetch(self, title, revid):
        self.fetched.append((title, revid))
        wikitext, expected = ARTICLES[title]
        self.assertEqual(revid, expected)
        return wikitext

    def load(self, **kwargs):
        return load_corpus(self.cache, manifest=self.manifest, fetch=self.fetch, **kwargs)


class LoadTest(CorpusTestCase):
    def test_downloads_once_then_offline_works(self):
        first = self.load()
        self.assertEqual(len(self.fetched), 3)
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()),
                         ["11.txt", "22.txt", "33.txt"])
        second = self.load(offline=True)
        self.assertEqual(len(self.fetched), 3)
        self.assertEqual(first, second)

    def test_text_content_and_order(self):
        text = self.load()
        self.assertEqual(text, "\n\n".join(
            f"{t}\n\n{self.plain[t]}" for t in ARTICLES))
        self.assertIn("Alpha is first.", text)

    def test_cache_is_reused_online(self):
        self.load()
        self.load()
        self.assertEqual(len(self.fetched), 3)

    def test_partial_cache_only_fetches_missing(self):
        self.load()
        (self.cache / "22.txt").unlink()
        self.fetched.clear()
        self.load()
        self.assertEqual(self.fetched, [("Beta River", 22)])

    def test_offline_with_empty_cache_names_all_articles(self):
        with self.assertRaises(CorpusError) as raised:
            self.load(offline=True)
        message = str(raised.exception)
        for title in ARTICLES:
            self.assertIn(title, message)
        self.assertEqual(self.fetched, [])

    def test_offline_incomplete_cache_names_missing_only(self):
        self.load()
        (self.cache / "22.txt").unlink()
        with self.assertRaises(CorpusError) as raised:
            self.load(offline=True)
        message = str(raised.exception)
        self.assertIn("Beta River", message)
        self.assertNotIn("Alpha", message)
        self.assertNotIn("Gamma", message)

    def test_checksum_mismatch(self):
        self.load()
        (self.cache / "22.txt").write_bytes(b"tampered")
        for offline in (False, True):
            with self.subTest(offline=offline):
                with self.assertRaises(ValueError) as raised:
                    self.load(offline=offline)
                self.assertIn("Beta River", str(raised.exception))
                self.assertIn("delete", str(raised.exception))
                self.assertIn("22.txt", str(raised.exception))

    def test_atomic_write_survives_a_transient_lock(self):
        real = os.replace
        calls = []

        def replace(src, dst):
            calls.append(1)
            if len(calls) == 1:
                raise PermissionError(13, "in use")
            return real(src, dst)

        with mock.patch("os.replace", replace), mock.patch("time.sleep"):
            self.load()
        self.assertGreater(len(calls), 1)
        self.assertTrue((self.cache / "11.txt").exists())

    def test_checksum_mismatch_on_download_caches_nothing(self):
        self.write_manifest({"Gamma": "0" * 64})
        with self.assertRaises(ValueError) as raised:
            self.load()
        self.assertIn("Gamma", str(raised.exception))
        self.assertFalse((self.cache / "33.txt").exists())
        # Articles before the bad one are fine and stay cached.
        self.assertTrue((self.cache / "11.txt").exists())

    def test_no_temp_files_left(self):
        self.load()
        self.assertEqual([p for p in self.cache.iterdir() if p.suffix != ".txt"], [])

    def test_cache_files_are_utf8_with_unix_newlines(self):
        self.load()
        data = (self.cache / "11.txt").read_bytes()
        self.assertEqual(data, self.plain["Alpha"].encode("utf-8"))
        self.assertNotIn(b"\r", data)

    def test_non_ascii_roundtrip(self):
        articles = dict(ARTICLES)
        articles["Gamma"] = ("Zürich – café 中", 33)
        with mock.patch.dict(ARTICLES, articles):
            self.plain["Gamma"] = wikitext_to_text(articles["Gamma"][0])
            self.write_manifest()
            first = self.load()
            self.assertEqual(self.load(offline=True), first)
            self.assertIn("Zürich – café", first)


class ManifestTest(CorpusTestCase):
    def test_malformed_manifest(self):
        bad = {
            "no articles": "",
            "missing sha": '[[article]]\ntitle = "A"\nrevid = 1\n',
            "bad sha": '[[article]]\ntitle = "A"\nrevid = 1\nsha256 = "xyz"\n',
            "bad revid": '[[article]]\ntitle = "A"\nrevid = "1"\nsha256 = "%s"\n' % ("0" * 64),
            "duplicate": ('[[article]]\ntitle = "A"\nrevid = 1\nsha256 = "%s"\n' % ("0" * 64)) * 2,
        }
        for name, content in bad.items():
            with self.subTest(name=name):
                self.manifest.write_text(content, encoding="utf-8")
                with self.assertRaises(CorpusError):
                    self.load(offline=True)

    def test_shipped_manifest(self):
        data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
        articles = data["article"]
        self.assertGreaterEqual(len(articles), 17)
        titles = [a["title"] for a in articles]
        for expected in ("Rhine", "Danube", "Elbe", "Hamburg", "Munich"):
            self.assertIn(expected, titles)
        self.assertEqual(len(titles), len(set(titles)))
        for article in articles:
            self.assertIsInstance(article["revid"], int)
            self.assertRegex(article["sha256"], r"^[0-9a-f]{64}$")
        # The loader accepts the shipped manifest as is.
        self.assertEqual(len(corpus.read_manifest(MANIFEST)), len(articles))

    def test_default_manifest_is_the_shipped_one(self):
        self.assertEqual(corpus.DEFAULT_MANIFEST, MANIFEST)

    def test_no_corpus_text_in_manifest(self):
        self.assertLess(MANIFEST.stat().st_size, 20000)


class PinTest(unittest.TestCase):
    def test_render_manifest(self):
        pinned = [corpus.Article("A", 5, "a" * 64), corpus.Article("B", 6, "b" * 64)]
        text = corpus.render_manifest(pinned)
        data = tomllib.loads(text)
        self.assertEqual([a["title"] for a in data["article"]], ["A", "B"])
        self.assertEqual(data["article"][1]["revid"], 6)
        self.assertIn("CC BY-SA", text)
        self.assertNotIn("\u2014", text)

    def test_pin_article_uses_current_revision_once(self):
        def latest(title):
            return 99

        def fetch(title, revid):
            self.assertEqual(revid, 99)
            return "Hello [[world]]."

        pinned = corpus.pin_article("A", latest=latest, fetch=fetch)
        self.assertEqual(pinned, corpus.Article("A", 99, sha("Hello world.")))

    def test_pin_rejects_redirects(self):
        with self.assertRaisesRegex(CorpusError, "redirect"):
            corpus.pin_article("Thames", latest=lambda t: 5,
                               fetch=lambda t, r: "#REDIRECT [[River Thames]]")


@unittest.skipUnless(os.environ.get("BENCH_LIVE") == "1", "set BENCH_LIVE=1 to download")
class LiveTest(unittest.TestCase):
    def test_real_download_matches_pinned_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "corpus"
            text = load_corpus(cache)
            self.assertGreater(len(text), 1_500_000)
            self.assertEqual(load_corpus(cache, offline=True), text)


if __name__ == "__main__":
    unittest.main()
