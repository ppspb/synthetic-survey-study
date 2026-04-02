from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = [
    ROOT / "data" / "mock" / "samples.tsv",
    ROOT / "data" / "mock" / "question_metadata.json",
]
private_checks = [
    ROOT / "data" / "private" / "wvs_wave7_source.zip",
    ROOT / "data" / "private" / "samples.tsv",
]
print("Public files:")
for p in checks:
    print(f"[{'ok' if p.exists() else 'missing'}] {p.relative_to(ROOT)}")
print("\nPrivate reproduction files:")
for p in private_checks:
    print(f"[{'ok' if p.exists() else 'missing'}] {p.relative_to(ROOT)}")
print("\nIf private files are missing, run:")
print("python scripts/prepare_wvs_wave7.py --source-zip <path-to-downloaded-wvs-wave7-zip>")
