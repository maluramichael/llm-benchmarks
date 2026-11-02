import json
import unittest
from pathlib import Path
from unittest import mock

from bench import privacy
from bench.privacy import find_leaks, local_secrets, scrub

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SECRETS = ["nero", "peon", "michael"]


class ScrubTest(unittest.TestCase):
    def test_scrub_paths(self):
        self.assertEqual(scrub(r"C:\Users\michael\x", ["michael"]), "<home>\\x")
        self.assertEqual(scrub("/home/michael/x", ["michael"]), "<home>/x")
        self.assertEqual(scrub("/Users/michael/x", ["michael"]), "<home>/x")

    def test_scrub_paths_without_known_secret(self):
        self.assertEqual(scrub(r"C:\Users\someone\x", []), "<home>\\x")
        self.assertEqual(scrub("/home/someone/x", []), "<home>/x")

    def test_scrub_json_escaped_windows_path(self):
        self.assertEqual(scrub("C:\\\\Users\\\\michael\\\\x", ["michael"]), "<home>\\\\x")

    def test_scrub_home_path_with_space(self):
        text = r"open C:\Users\Max Mustermann\file.txt failed"
        self.assertEqual(scrub(text, [r"C:\Users\Max Mustermann"]), "open <home>\\file.txt failed")

    def test_scrub_shared_profiles_are_kept(self):
        self.assertEqual(scrub(r"C:\Users\Public\x", []), r"C:\Users\Public\x")
        self.assertEqual(scrub("/Users/Shared/x", []), "/Users/Shared/x")

    def test_scrub_secret_words(self):
        self.assertEqual(scrub("host NERO and peon-1", ["nero", "peon"]),
                         "host <redacted> and <redacted>-1")

    def test_scrub_keeps_words_that_merely_contain_a_secret(self):
        self.assertEqual(scrub("Nerone and openeron", ["nero"]), "Nerone and openeron")

    def test_scrub_ip_and_mac(self):
        self.assertEqual(scrub("ip 192.168.2.10 mac aa:bb:cc:dd:ee:ff", []),
                         "ip <ip> mac <mac>")

    def test_scrub_keeps_version_numbers(self):
        text = "driver 32.0.15.9159 build 10.0.26100 kernel 6.12.101"
        self.assertEqual(scrub(text, []), text)

    def test_scrub_nested_objects(self):
        obj = {"a": ["/home/michael/x", 3, None, True], "nero": {"b": ("peon", 1.5)}}
        self.assertEqual(
            scrub(obj, SECRETS),
            {"a": ["<home>/x", 3, None, True], "<redacted>": {"b": ("<redacted>", 1.5)}})

    def test_scrub_ignores_empty_secrets(self):
        self.assertEqual(scrub("abc", ["", None]), "abc")

    def test_scrub_is_idempotent(self):
        once = scrub("/home/michael/x on nero 10.1.2.3", SECRETS)
        self.assertEqual(scrub(once, SECRETS), once)


class FindLeaksTest(unittest.TestCase):
    def test_find_leaks_ip_host(self):
        leaks = find_leaks("server nero at 192.168.2.10 answered", ["nero"])
        self.assertTrue(any("192.168.2.10" in leak for leak in leaks), leaks)
        self.assertTrue(any("nero" in leak for leak in leaks), leaks)

    def test_find_leaks_clean(self):
        text = "RTX 3090, 24576 MiB, driver 591.59, Windows 10.0.26100, kernel 6.12.101"
        self.assertEqual(find_leaks(text, SECRETS), [])

    def test_find_leaks_user_path_of_unknown_user(self):
        leaks = find_leaks(r'File "C:\Users\somebody\x.py", line 3', [])
        self.assertEqual(len(leaks), 1, leaks)
        self.assertIn("somebody", leaks[0])
        self.assertTrue(find_leaks("/home/somebody/x.py", []))
        self.assertTrue(find_leaks("/Users/somebody/x.py", []))

    def test_find_leaks_scrubbed_text_is_clean(self):
        dirty = r"File C:\Users\somebody\x.py on nero 10.0.0.7 aa:bb:cc:dd:ee:ff fe80::1a2b"
        self.assertTrue(find_leaks(dirty, ["nero"]))
        self.assertEqual(find_leaks(scrub(dirty, ["nero"]), ["nero"]), [])

    def test_find_leaks_mac_and_ipv6(self):
        self.assertTrue(find_leaks("aa:bb:cc:dd:ee:ff", []))
        self.assertTrue(find_leaks("link fe80::1a2b:3c4d", []))

    def test_find_leaks_invalid_ip_is_not_a_leak(self):
        self.assertEqual(find_leaks("version 300.1.2.3", []), [])

    def test_find_leaks_case_insensitive_and_deduplicated(self):
        leaks = find_leaks("NERO nero Nero", ["nero"])
        self.assertEqual(len(leaks), 1, leaks)

    def test_find_leaks_secret_is_not_matched_inside_words(self):
        self.assertEqual(find_leaks("Nerone", ["nero"]), [])

    def test_committed_fixtures_are_clean(self):
        files = sorted(p for p in FIXTURES.iterdir() if p.is_file())
        self.assertGreater(len(files), 5)
        for path in files:
            with self.subTest(fixture=path.name):
                self.assertEqual(find_leaks(path.read_text(encoding="utf-8"), SECRETS), [])


class LocalSecretsTest(unittest.TestCase):
    def test_contains_host_user_and_home_in_both_spellings(self):
        with mock.patch.object(privacy.socket, "gethostname", return_value="Nero.example.lan"), \
                mock.patch.object(privacy.getpass, "getuser", return_value="michael"), \
                mock.patch.object(privacy.Path, "home", return_value=Path("/home/michael")), \
                mock.patch.dict(privacy.os.environ, {"USERNAME": "WinMichael"}, clear=True):
            secrets = local_secrets()
        for expected in ["Nero.example.lan", "Nero", "michael", "WinMichael",
                         "/home/michael", "\\home\\michael"]:
            self.assertIn(expected, secrets)

    def test_windows_home(self):
        with mock.patch.object(privacy.socket, "gethostname", return_value="nero"), \
                mock.patch.object(privacy.getpass, "getuser", return_value="michael"), \
                mock.patch.object(privacy.Path, "home",
                                  return_value=mock.Mock(__str__=lambda s: "C:\\Users\\michael")), \
                mock.patch.dict(privacy.os.environ, {}, clear=True):
            secrets = local_secrets()
        self.assertIn("C:\\Users\\michael", secrets)
        self.assertIn("C:/Users/michael", secrets)

    def test_never_raises_and_has_no_empty_entries(self):
        with mock.patch.object(privacy.socket, "gethostname", side_effect=OSError), \
                mock.patch.object(privacy.getpass, "getuser", side_effect=KeyError):
            secrets = local_secrets()
        self.assertIsInstance(secrets, list)
        self.assertNotIn("", secrets)

    def test_scrubbing_with_local_secrets_removes_them(self):
        secrets = local_secrets()
        text = json.dumps({"p": str(Path.home()) + "/x"})
        self.assertEqual(find_leaks(scrub(text, secrets), secrets), [])


if __name__ == "__main__":
    unittest.main()
