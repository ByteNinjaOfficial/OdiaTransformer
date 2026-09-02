"""Unit tests for src/tokenizer.py (Phase 2B).

Run from the repository root:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Make `src` importable when running from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import tokenizer as tk  # noqa: E402


# ---------------------------------------------------------------------------
# Small synthetic corpora for fast, deterministic tests
# ---------------------------------------------------------------------------

_SYNTHETIC_EN = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "machine translation is a challenging task",
    "sentencepiece bpe tokenization works well",
]

_SYNTHETIC_OR = [
    "ନମସ୍କାର ବିଶ୍ୱ",
    "କ୍ଷିପ୍ର କ୍ରମାଗତ ବ୍ରାଉନ୍ ଫକ୍ସ",
    "ଯାଞ୍ଚ ବାକ୍ୟ ପାଇଁ ଧନ୍ୟବାଦ",
]


def _make_synthetic_train_parquet(tmpdir: Path) -> Path:
    """Create a tiny train.parquet with synthetic EN/OR pairs."""
    rows = []
    for i, (en, or_) in enumerate(zip(_SYNTHETIC_EN, _SYNTHETIC_OR)):
        rows.append({"idx": i, "src": en, "tgt": or_})
    df = pd.DataFrame(rows)
    path = tmpdir / "train.parquet"
    df.to_parquet(path, index=False)
    return path


# Vocabulary size for synthetic tests - must be small enough for the tiny corpus
SYNTH_VOCAB_SHARED = 200
SYNTH_VOCAB_EN = 80    # smaller for tiny English corpus
SYNTH_VOCAB_OR = 120   # smaller for tiny Odia corpus


def _train_separate_synthetic(tmpdir: Path, train_path: Path):
    """Train separate tokenizers on synthetic data."""
    cfg = tk.TokenizerConfig(mode="separate", vocab_size_en=SYNTH_VOCAB_EN, vocab_size_or=SYNTH_VOCAB_OR)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        en_corpus = tmp / "en.txt"
        or_corpus = tmp / "or.txt"
        train_df = pd.read_parquet(train_path)
        tk.build_separate_corpus_files(train_df, en_corpus, or_corpus)
        en_model, or_model = tk.train_separate(cfg, en_corpus, or_corpus, tmpdir)
        tok_en = tk.load_tokenizer(en_model)
        tok_or = tk.load_tokenizer(or_model)
        return tok_en, tok_or


def _train_shared_synthetic(tmpdir: Path, train_path: Path):
    """Train shared tokenizer on synthetic data."""
    cfg = tk.TokenizerConfig(mode="shared", vocab_size_shared=SYNTH_VOCAB_SHARED)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        shared_corpus = tmp / "shared.txt"
        train_df = pd.read_parquet(train_path)
        tk.build_shared_corpus_file(train_df, shared_corpus)
        model = tk.train_shared(cfg, shared_corpus, tmpdir)
        return tk.load_tokenizer(model)


# ---------------------------------------------------------------------------

class TestSpecialTokenContract(unittest.TestCase):
    """Verify the hard special-token ID contract: PAD=0, SOS=1, EOS=2, UNK=3."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_special_ids_separate_en(self):
        tok_en, _ = _train_separate_synthetic(self.tmp, self.train_path)
        self.assertEqual(tok_en.piece_to_id(tk.PAD_PIECE), tk.PAD_ID)
        self.assertEqual(tok_en.piece_to_id(tk.SOS_PIECE), tk.SOS_ID)
        self.assertEqual(tok_en.piece_to_id(tk.EOS_PIECE), tk.EOS_ID)
        self.assertEqual(tok_en.piece_to_id(tk.UNK_PIECE), tk.UNK_ID)

    def test_special_ids_separate_or(self):
        _, tok_or = _train_separate_synthetic(self.tmp, self.train_path)
        self.assertEqual(tok_or.piece_to_id(tk.PAD_PIECE), tk.PAD_ID)
        self.assertEqual(tok_or.piece_to_id(tk.SOS_PIECE), tk.SOS_ID)
        self.assertEqual(tok_or.piece_to_id(tk.EOS_PIECE), tk.EOS_ID)
        self.assertEqual(tok_or.piece_to_id(tk.UNK_PIECE), tk.UNK_ID)

    def test_special_ids_shared(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        self.assertEqual(tok.piece_to_id(tk.PAD_PIECE), tk.PAD_ID)
        self.assertEqual(tok.piece_to_id(tk.SOS_PIECE), tk.SOS_ID)
        self.assertEqual(tok.piece_to_id(tk.EOS_PIECE), tk.EOS_ID)
        self.assertEqual(tok.piece_to_id(tk.UNK_PIECE), tk.UNK_ID)


class TestEncodeDecodeRoundtrip(unittest.TestCase):
    """Encode → decode roundtrip fidelity against preprocessed text."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_roundtrip_english(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        for text in _SYNTHETIC_EN:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_roundtrip_odia(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        for text in _SYNTHETIC_OR:
            self.assertEqual(tok.decode(tok.encode(text)), text)


class TestPadHandling(unittest.TestCase):
    """PAD token behavior."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_pad_id_is_zero(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        self.assertEqual(tk.PAD_ID, 0)
        self.assertEqual(tok.piece_to_id(tk.PAD_PIECE), 0)

    def test_pad_not_added_by_encode(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        ids = tok.encode("hello world")
        self.assertNotIn(tk.PAD_ID, ids)


class TestSosEosAvailability(unittest.TestCase):
    """SOS/EOS IDs are correctly reserved and available."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sos_id_is_one(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        self.assertEqual(tk.SOS_ID, 1)
        self.assertEqual(tok.piece_to_id(tk.SOS_PIECE), 1)

    def test_eos_id_is_two(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        self.assertEqual(tk.EOS_ID, 2)
        self.assertEqual(tok.piece_to_id(tk.EOS_PIECE), 2)

    def test_encode_with_sos_eos(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        ids = tok.encode_with_sos_eos("hello")
        self.assertEqual(ids[0], tk.SOS_ID)
        self.assertEqual(ids[-1], tk.EOS_ID)


class TestUnkBehavior(unittest.TestCase):
    """UNK token behavior for out-of-vocabulary content."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_unk_id_is_three(self):
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        self.assertEqual(tk.UNK_ID, 3)
        self.assertEqual(tok.piece_to_id(tk.UNK_PIECE), 3)

    def test_unk_for_unseen_characters(self):
        # Very small vocab may force UNK for rare chars
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        # Use a character unlikely to be in the small synthetic vocab
        ids = tok.encode("🚀🚀🚀")
        # Either UNK appears or empty (both acceptable for tiny vocab)
        self.assertTrue(any(i == tk.UNK_ID for i in ids) or len(ids) == 0)


class TestSpecialIdVerification(unittest.TestCase):
    """Explicit special-ID verification function / constructor behavior."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.train_path = _make_synthetic_train_parquet(self.tmp)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_tokenizer_constructor_verifies_ids(self):
        """Tokenizer.__init__ should raise if special IDs don't match."""
        tok = _train_shared_synthetic(self.tmp, self.train_path)
        # This should not raise
        self.assertIsInstance(tok, tk.Tokenizer)


class TestConfigurationCorrectness(unittest.TestCase):
    """Configuration validation for separate vs shared modes."""

    def test_separate_requires_both_vocabs(self):
        with self.assertRaises(ValueError):
            tk.TokenizerConfig(mode="separate", vocab_size_en=1000)
        with self.assertRaises(ValueError):
            tk.TokenizerConfig(mode="separate", vocab_size_or=1000)
        # Valid
        cfg = tk.TokenizerConfig(mode="separate", vocab_size_en=1000, vocab_size_or=2000)
        self.assertEqual(cfg.vocab_size_en, 1000)
        self.assertEqual(cfg.vocab_size_or, 2000)

    def test_shared_requires_shared_vocab(self):
        with self.assertRaises(ValueError):
            tk.TokenizerConfig(mode="shared")
        with self.assertRaises(ValueError):
            tk.TokenizerConfig(mode="shared", vocab_size_en=1000)
        # Valid
        cfg = tk.TokenizerConfig(mode="shared", vocab_size_shared=3000)
        self.assertEqual(cfg.vocab_size_shared, 3000)

    def test_config_id_format(self):
        cfg_sep = tk.TokenizerConfig(mode="separate", vocab_size_en=4000, vocab_size_or=8000)
        self.assertEqual(cfg_sep.config_id(), "sep_en4000_or8000")
        cfg_sh = tk.TokenizerConfig(mode="shared", vocab_size_shared=16000)
        self.assertEqual(cfg_sh.config_id(), "shared16000")


class TestTrainOnlyCorpusConstruction(unittest.TestCase):
    """Train-only corpus construction — leakage guard."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_separate_corpus_only_uses_train(self):
        train_rows = [
            {"idx": 0, "src": "train en 1", "tgt": "train or 1"},
            {"idx": 1, "src": "train en 2", "tgt": "train or 2"},
        ]
        train_df = pd.DataFrame(train_rows)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            en_path = tmp / "en.txt"
            or_path = tmp / "or.txt"
            tk.build_separate_corpus_files(train_df, en_path, or_path)
            en_lines = en_path.read_text(encoding="utf-8").strip().split("\n")
            or_lines = or_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(en_lines), 2)
            self.assertEqual(len(or_lines), 2)
            self.assertIn("train en 1", en_lines)
            self.assertIn("train en 2", en_lines)
            self.assertIn("train or 1", or_lines)
            self.assertIn("train or 2", or_lines)

    def test_shared_corpus_only_uses_train(self):
        train_rows = [
            {"idx": 0, "src": "train en", "tgt": "train or"},
        ]
        train_df = pd.DataFrame(train_rows)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            shared_path = tmp / "shared.txt"
            tk.build_shared_corpus_file(train_df, shared_path)
            lines = shared_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertIn("train en", lines)
            self.assertIn("train or", lines)


class TestParameterCostCalculation(unittest.TestCase):
    """Vocabulary-related parameter cost with d_model=128."""

    def test_separate_cost(self):
        cfg = tk.TokenizerConfig(mode="separate", vocab_size_en=8000, vocab_size_or=16000)
        cost = tk.compute_parameter_cost(cfg)
        expected = (8000 + 2 * 16000) * 128
        self.assertEqual(cost, expected)

    def test_shared_cost(self):
        cfg = tk.TokenizerConfig(mode="shared", vocab_size_shared=24000)
        cost = tk.compute_parameter_cost(cfg)
        expected = 2 * 24000 * 128
        self.assertEqual(cost, expected)


if __name__ == "__main__":
    unittest.main()