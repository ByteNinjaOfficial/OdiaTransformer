"""Run the Phase 1C preprocessing pipeline against the real Samanantar Odia parquet."""
import sys, os
sys.path.insert(0, os.path.abspath("src"))

import preprocessing as pp
from pathlib import Path

RAW = Path("dataset/train-00000-of-00001.parquet")
OUT = Path("outputs")

res = pp.pipeline(RAW, OUT, split=pp.SplitConfig(0.98, 0.01, seed=42))
print("PIPELINE OK")
print("raw_rows:", res.report["raw_input"]["rows"])
print("raw_immutable:", res.report["raw_input"]["raw_immutable"])
print("raw_sha256_before:", res.report["raw_input"]["sha256_before"][:16])
print("raw_sha256_after :", res.report["raw_input"]["sha256_after"][:16])
print("rows_after_each_step:", res.report["rows_after_each_step"])
print("nfc before:", res.report["nfc"]["before"])
print("leading_quote_cleaned:", res.report["leading_quote_artifact_cleaned"])
print("dup removed:", res.report["duplicates"]["removed"])
print("numeric:", res.report["numeric_noise"])
print("split:", res.report["split"])
print("report_path:", res.report["report_path"])
print("train", res.train.shape, "val", res.val.shape, "test", res.test.shape)
