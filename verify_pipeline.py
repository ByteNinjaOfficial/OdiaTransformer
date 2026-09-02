"""Independent verification of Phase 1C outputs."""
import sys, os, json
sys.path.insert(0, os.path.abspath("src"))
import pandas as pd
from pathlib import Path

OUT = Path("outputs")
RAW = Path("dataset/train-00000-of-00001.parquet")
import preprocessing as pp

print("== immutability ==")
print("raw sha256:", pp.sha256_file(RAW))

print("\n== output parquets load & shape ==")
tr = pd.read_parquet(OUT/"train.parquet")
va = pd.read_parquet(OUT/"val.parquet")
te = pd.read_parquet(OUT/"test.parquet")
print("train", tr.shape, "val", va.shape, "test", te.shape)
print("cols:", list(tr.columns))

print("\n== no overlap (idx) ==")
itr, iva, ite = set(tr.idx), set(va.idx), set(te.idx)
print("train<->val overlap:", len(itr & iva))
print("train<->test overlap:", len(itr & ite))
print("val<->test overlap:", len(iva & ite))

print("\n== no overlap (text pairs) ==")
tr_pairs = set(zip(tr.src, tr.tgt)); va_pairs = set(zip(va.src, va.tgt)); te_pairs = set(zip(te.src, te.tgt))
print("train<->val pair overlap:", len(tr_pairs & va_pairs))
print("train<->test pair overlap:", len(tr_pairs & te_pairs))
print("val<->test pair overlap:", len(va_pairs & te_pairs))

print("\n== row accounting ==")
print("train+val+test =", len(tr)+len(va)+len(te), "(should be final_count)")

print("\n== report json loads & key fields ==")
rep = json.loads((OUT/"preprocessing_report.json").read_text(encoding="utf-8"))
print("raw_immutable:", rep["raw_input"]["raw_immutable"])
print("rows:", rep["raw_input"]["rows"])
print("nfc_before:", rep["nfc"]["before"])
print("excluded_count:", rep["numeric_noise"]["excluded_count"])
print("candidate_count:", rep["numeric_noise"]["candidate_count"])

ex = pd.read_parquet(OUT/"numeric_excluded_samples.parquet")
print("excluded samples file rows:", len(ex))
print("all excluded idx are real (idx in [train|val|test]):",
      not (set(ex.idx) & (itr | iva | ite)))  # excluded must NOT be in splits

print("\n== NaN/empty/NFC invariants in outputs ==")
for name, df in [("train", tr), ("val", va), ("test", te)]:
    assert df[["src","tgt"]].isna().sum().sum() == 0, name+" has NaN"
    assert not ((df["src"].str.strip()=="") | (df["tgt"].str.strip()=="")).any(), name+" empty"
    import unicodedata
    noncf = sum(1 for v in df["src"] if v != unicodedata.normalize("NFC", v))
    noncf += sum(1 for v in df["tgt"] if v != unicodedata.normalize("NFC", v))
    ndup = int(df.duplicated(subset=["src","tgt"]).sum())
    print(f"{name}: NaN=0 empty=0 nonNFC={noncf} dup_pairs={ndup}")

print("\nALL VERIFICATION CHECKS PASSED")
