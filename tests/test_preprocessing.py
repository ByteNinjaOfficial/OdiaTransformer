"""Unit tests for src/preprocessing.py (Phase 1C).

Run from the repository root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import hashlib
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

import pandas as pd

# Make `src` importable when running from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import preprocessing as pp  # noqa: E402


def make_df(rows):
    """Build a DataFrame with the expected schema from a list of (idx, src, tgt)."""
    return pd.DataFrame(rows, columns=["idx", "src", "tgt"])


# A genuinely non-normalized Odia string: `େ` (0B47) + `ା` (0B3E), which NFC
# composes to `ୋ` (0B4B). Built via unicodedata so we never hardcode a literal
# that may already be normalized.
_ODIA_NFD = "\u0B47\u0B3E"
_ODIA_NFC = unicodedata.normalize("NFC", _ODIA_NFD)


class TestSchemaValidation(unittest.TestCase):
    def test_accepts_correct_schema(self):
        df = make_df([(0, "a", "b")])
        self.assertIs(pp.validate_schema(df), df)

    def test_rejects_wrong_column_order(self):
        df = pd.DataFrame({"src": ["a"], "tgt": ["b"], "idx": [0]})
        with self.assertRaises(ValueError):
            pp.validate_schema(df)

    def test_rejects_missing_column(self):
        df = pd.DataFrame({"idx": [0], "src": ["a"]})
        with self.assertRaises(ValueError):
            pp.validate_schema(df)

    def test_rejects_non_integer_idx(self):
        df = pd.DataFrame({"idx": [0.0], "src": ["a"], "tgt": ["b"]})
        with self.assertRaises(ValueError):
            pp.validate_schema(df)


class TestTextValidation(unittest.TestCase):
    def test_clean_passes(self):
        df = make_df([(0, "hello", "ନମସ୍କାର")])
        summary = pp.validate_text(df)
        self.assertEqual(summary["nulls"]["src"], 0)
        self.assertEqual(summary["empty"]["src"], 0)

    def test_raises_on_null(self):
        df = make_df([(0, None, "b")])
        with self.assertRaises(ValueError):
            pp.validate_text(df)

    def test_raises_on_empty_string(self):
        df = make_df([(0, "  ", "b")])
        with self.assertRaises(ValueError):
            pp.validate_text(df)


class TestNFC(unittest.TestCase):
    def test_normalizes_both_src_and_tgt(self):
        df = make_df([(0, _ODIA_NFD, _ODIA_NFD)])
        out = pp.normalize_nfc(df)
        self.assertEqual(out.loc[0, "src"], _ODIA_NFC)
        self.assertEqual(out.loc[0, "tgt"], _ODIA_NFC)

    def test_count_non_nfc(self):
        df = make_df([(0, "plain", _ODIA_NFD), (1, "x", "ଓଡ଼ିଆ")])
        self.assertEqual(pp.count_non_nfc(df["tgt"]), 1)
        self.assertEqual(pp.count_non_nfc(df["src"]), 0)

    def test_nfc_idempotent(self):
        once = pp.apply_nfc(_ODIA_NFD)
        self.assertEqual(once, pp.apply_nfc(once))

    def test_no_nfkc_compat_mapping(self):
        # A fullwidth ASCII digit (FF11) is NFC-stable but NFKC would map it to
        # the ASCII '1'. Since we apply only NFC, it must be preserved as-is.
        self.assertEqual(pp.apply_nfc("\uFF11"), "\uFF11")


class TestLeadingQuoteCleanup(unittest.TestCase):
    def test_strips_leading_triple_quote(self):
        self.assertEqual(pp.clean_source_leading_quote('"""He has worked."'), 'He has worked."')
        self.assertEqual(pp.clean_source_leading_quote('"""Modi ji spoke.'), 'Modi ji spoke.')

    def test_preserves_single_leading_quote(self):
        s = '"Normal quoted sentence."'
        self.assertEqual(pp.clean_source_leading_quote(s), s)

    def test_preserves_double_leading_quote_nested(self):
        # `""` leading is a legitimate nested-quote pattern; preserved.
        s = '""Defence "Team" target'
        self.assertEqual(pp.clean_source_leading_quote(s), s)

    def test_preserves_no_quote(self):
        self.assertEqual(pp.clean_source_leading_quote("hello world"), "hello world")

    def test_cleanup_only_touches_src(self):
        df = make_df([(0, '"""He came.', "ସେ ଆସିଲେ।")])
        out = pp.apply_cleanup(df)
        self.assertEqual(out.loc[0, "src"], "He came.")
        self.assertEqual(out.loc[0, "tgt"], "ସେ ଆସିଲେ।")


class TestDedup(unittest.TestCase):
    def test_removes_exact_pairs(self):
        df = make_df([(0, "a", "x"), (1, "a", "x")])
        out, removed = pp.deduplicate_pairs(df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(out), 1)

    def test_preserves_duplicate_src_different_tgt(self):
        df = make_df([(0, "a", "x"), (1, "a", "y")])
        out, removed = pp.deduplicate_pairs(df)
        self.assertEqual(removed, 0)
        self.assertEqual(len(out), 2)

    def test_preserves_duplicate_tgt(self):
        df = make_df([(0, "a", "x"), (1, "b", "x")])
        out, removed = pp.deduplicate_pairs(df)
        self.assertEqual(removed, 0)
        self.assertEqual(len(out), 2)


class TestNumericDefinition(unittest.TestCase):
    def test_numeric_token_definition(self):
        self.assertTrue("417.44".replace(".", "").replace(",", "").isdigit())
        self.assertTrue("1,234".replace(".", "").replace(",", "").isdigit())
        self.assertTrue("7".replace(".", "").replace(",", "").isdigit())
        self.assertFalse("n1".replace(".", "").replace(",", "").isdigit())
        self.assertFalse("Rajasthan".replace(".", "").replace(",", "").isdigit())

    def test_fraction(self):
        self.assertEqual(pp.compute_numeric_fraction("1 2 3"), 1.0)
        self.assertEqual(pp.compute_numeric_fraction("a 1 2"), 2 / 3)
        self.assertEqual(pp.compute_numeric_fraction("a b c"), 0.0)
        self.assertEqual(pp.compute_numeric_fraction(""), 0.0)

    def test_strict_greater_than_threshold(self):
        # 7 numeric tokens + 13 words = 20 tokens -> exactly 0.35, NOT candidate.
        ex = " ".join(["1", "2", "3", "4", "5", "6", "7"] + ["w"] * 13)
        self.assertEqual(pp.compute_numeric_fraction(ex), 7 / 20)
        self.assertFalse(pp.numeric_candidate(ex, "a b c", 0.35))
        # 8 numeric + 13 words = 21 -> 8/21 ~ 0.381 > 0.35 -> candidate.
        ex2 = " ".join(["1", "2", "3", "4", "5", "6", "7", "8"] + ["w"] * 13)
        self.assertGreater(pp.compute_numeric_fraction(ex2), 0.35)
        self.assertTrue(pp.numeric_candidate(ex2, "a b c", 0.35))


class TestConservativeNumericFilter(unittest.TestCase):
    def test_table_both_sides_excluded(self):
        # Genuinely long malformed spreadsheet table: numeric-heavy on BOTH
        # sides with >= 20 source tokens -> excluded.
        src = " ".join([str(i) for i in range(1, 21)]) + " Rajasthan 417.44"
        tgt = " ".join([str(i) for i in range(1, 21)])
        df = make_df([(0, src, tgt)])
        out, idx, info = pp.numeric_noise_filter(df)
        self.assertEqual(info["excluded_count"], 1)
        self.assertEqual(info["candidate_count"], 1)
        self.assertEqual(len(out), 0)

    def test_short_both_sides_numeric_kept(self):
        # "13 places" is a legitimate short translation, numeric-heavy on both
        # sides but only 2 source tokens -> preserved, never deleted.
        df = make_df([(0, "13 places", "୧୩ ସ୍ଥାନଗୁଡ଼ିକ")])
        out, idx, info = pp.numeric_noise_filter(df)
        self.assertGreaterEqual(info["candidate_count"], 1)
        self.assertEqual(info["excluded_count"], 0)
        self.assertEqual(len(out), 1)

    def test_table_length_floor_constant(self):
        self.assertGreaterEqual(pp._MIN_TABLE_SRC_TOKENS, 10)

    def test_source_numeric_only_kept(self):
        # Numeric-heavy English with a real (word-y) Odia target is a candidate
        # but NOT excluded -> legit sentence preserved.
        df = make_df([
            (0, "1 2 3 4 5 6 7 8 9 people attended", "ପ୍ରାୟ ତିରିଶ ଲକ୍ଷ ଲୋକ")
        ])
        out, idx, info = pp.numeric_noise_filter(df)
        self.assertGreaterEqual(info["candidate_count"], 1)
        self.assertEqual(info["excluded_count"], 0)
        self.assertEqual(len(out), 1)

    def test_below_threshold_kept(self):
        df = make_df([(0, "one two three", "ଏକ ଦୁଇ ତିନି")])
        out, idx, info = pp.numeric_noise_filter(df)
        self.assertEqual(info["candidate_count"], 0)
        self.assertEqual(info["excluded_count"], 0)
        self.assertEqual(len(out), 1)


class TestNoLengthFilter(unittest.TestCase):
    def test_long_sentences_retained(self):
        long_src = " ".join(["word"] * 500)
        df = make_df([(0, long_src, "ଲମ୍ବା ବାକ୍ୟ")])
        out, idx, info = pp.numeric_noise_filter(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "src"], long_src)


class TestSplit(unittest.TestCase):
    def test_exact_accounting(self):
        df = make_df([(i, "a", "b") for i in range(1000)])
        tr, va, te = pp.split_dataset(df, 0.98, 0.01, seed=42)
        self.assertEqual(len(tr) + len(va) + len(te), 1000)

    def test_no_overlap(self):
        df = make_df([(i, "a", "b") for i in range(1000)])
        tr, va, te = pp.split_dataset(df, 0.98, 0.01, seed=1)
        itr, iva, ite = set(tr.idx), set(va.idx), set(te.idx)
        self.assertTrue(itr.isdisjoint(iva))
        self.assertTrue(itr.isdisjoint(ite))
        self.assertTrue(iva.isdisjoint(ite))

    def test_deterministic(self):
        df = make_df([(i, "a", "b") for i in range(1000)])
        tr1, _, _ = pp.split_dataset(df, 0.98, 0.01, seed=7)
        tr2, _, _ = pp.split_dataset(df, 0.98, 0.01, seed=7)
        pd.testing.assert_frame_equal(tr1, tr2)

    def test_seed_changes_split(self):
        df = make_df([(i, "a", "b") for i in range(1000)])
        tr1, _, _ = pp.split_dataset(df, 0.98, 0.01, seed=7)
        tr2, _, _ = pp.split_dataset(df, 0.98, 0.01, seed=8)
        self.assertFalse(set(tr1.idx) == set(tr2.idx))

    def test_rejects_invalid_fractions(self):
        df = make_df([(i, "a", "b") for i in range(10)])
        with self.assertRaises(ValueError):
            pp.split_dataset(df, 1.0, 0.0, seed=1)


class TestRawImmutability(unittest.TestCase):
    def test_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"hello world")
            self.assertEqual(pp.sha256_file(p), hashlib.sha256(b"hello world").hexdigest())

    def test_sha256_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"AAAA")
            h1 = pp.sha256_file(p)
            p.write_bytes(b"BBBB")
            h2 = pp.sha256_file(p)
            self.assertNotEqual(h1, h2)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "out"
            rows = [
                # row 0: leading triple-quote artifact + numeric candidate,
                # real Odia target -> cleaned, kept.
                (0, '"""He came today with 1 2 3 4 5 6 7 8 9 people', "ସେ ଆଜି ଆସିଲେ"),
                # row 1: genuinely malformed long table on both sides -> excluded.
                (1, "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
                 "25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"),
                # row 2: non-normalized Odia -> NFC composed.
                (2, "a normal sentence", _ODIA_NFD + " ବାକ୍ୟ"),
            ]
            raw_path = Path(d) / "raw.parquet"
            make_df(rows).to_parquet(raw_path, index=False)

            result = pp.pipeline(raw_path, out_dir, split=pp.SplitConfig(0.6, 0.2, seed=1))
            r = result.report

            # Raw input is untouched.
            self.assertTrue(r["raw_input"]["raw_immutable"])
            self.assertEqual(r["raw_input"]["rows"], 3)

            # NFC: one non-NFC target before, none after.
            self.assertEqual(r["nfc"]["before"]["tgt"], 1)
            self.assertEqual(r["nfc"]["after"]["tgt"], 0)
            self.assertEqual(r["nfc"]["before"]["src"], 0)

            # Leading triple-quote artifact was cleaned on row 0's src.
            self.assertEqual(r["leading_quote_artifact_cleaned"], 1)

            # Numeric noise: row 1 excluded, row 0 kept.
            self.assertEqual(r["numeric_noise"]["excluded_count"], 1)
            self.assertEqual(r["numeric_noise"]["candidate_count"], 2)
            self.assertEqual(len(result.excluded_samples), 1)

            # Row-count accounting across steps is internal and consistent.
            after = r["rows_after_each_step"]
            self.assertEqual(after["final"],
                             after["raw"] - r["duplicates"]["removed"]
                             - r["numeric_noise"]["excluded_count"])

            # Split exact + no overlap.
            self.assertTrue(r["split"]["accounting_exact"])
            train, val, test = result.train, result.val, result.test
            itr, iva, ite = set(train.idx), set(val.idx), set(test.idx)
            self.assertTrue(itr.isdisjoint(iva) and itr.isdisjoint(ite) and iva.isdisjoint(ite))

            # Output files written.
            self.assertTrue((out_dir / "train.parquet").exists())
            self.assertTrue((out_dir / "val.parquet").exists())
            self.assertTrue((out_dir / "test.parquet").exists())
            self.assertTrue((out_dir / "preprocessing_report.json").exists())
            self.assertTrue((out_dir / "numeric_excluded_samples.parquet").exists())


if __name__ == "__main__":
    unittest.main()
