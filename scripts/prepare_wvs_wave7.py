from __future__ import annotations
import argparse, json, shutil, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare local benchmark files from the official WVS Wave 7 zip.")
    p.add_argument("--source-zip", required=True, help="Path to the official WVS Wave 7 CSV zip downloaded manually.")
    p.add_argument("--out-dir", default=str(ROOT / "data" / "private"), help="Directory for local-only prepared files.")
    p.add_argument("--question-metadata", default=str(ROOT / "data" / "benchmark_spec" / "question_metadata.json"))
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source_zip).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)

    meta = json.loads(Path(args.question_metadata).read_text(encoding="utf-8"))
    question_ids = sorted(meta.keys())

    copied_zip = out_dir / "wvs_wave7_source.zip"
    if src != copied_zip:
        shutil.copy2(src, copied_zip)

    with zipfile.ZipFile(src, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found inside the provided zip.")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    required_demo = ["D_INTERVIEW", "Continent", "Education"]
    urban_candidates = ["Urban / Rural", "Urban_Rural", "Urban/Rural"]
    urban_col = next((c for c in urban_candidates if c in df.columns), None)
    if urban_col is None:
        raise KeyError(f"Could not find an urban/rural column. Tried: {urban_candidates}")

    missing = [c for c in required_demo if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in downloaded CSV: {missing}")

    available_questions = [q for q in question_ids if q in df.columns]
    if not available_questions:
        raise KeyError("None of the benchmark question IDs were found in the downloaded CSV.")

    keep = ["D_INTERVIEW", "Continent", urban_col, "Education"] + available_questions
    slim = df[keep].copy()
    long_df = slim.melt(
        id_vars=["D_INTERVIEW", "Continent", urban_col, "Education"],
        var_name="QUESTION_ID",
        value_name="HUMAN_SCORE",
    )
    long_df = long_df.dropna(subset=["HUMAN_SCORE"]).copy()
    long_df["PARTICIPANT_ID"] = long_df["D_INTERVIEW"].astype(str)
    long_df["Urban_Rural"] = long_df[urban_col]
    long_df = long_df[["QUESTION_ID", "PARTICIPANT_ID", "Continent", "Urban_Rural", "Education", "HUMAN_SCORE"]]
    samples_path = out_dir / "samples.tsv"
    long_df.to_csv(samples_path, sep="\t", index=False)

    print("Prepared local-only benchmark files:")
    print(" -", copied_zip)
    print(" -", samples_path)
    print("\nNext step:")
    print("python scripts/run_yaml.py --config configs/reproduce/distribution_direct_persona.yaml --model <your-model>")


if __name__ == "__main__":
    main()
