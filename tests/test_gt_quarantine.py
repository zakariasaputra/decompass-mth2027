"""Stage 1 guard: the provided match's ground truth must not leak into pipeline or tuning code.

Only src/sanity_check.py (report-only correctness check) may reference evaluation/gt.
"""

import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = ("possession_gt", "evaluation/gt", "evaluation\\gt", '"gt"', "'gt'")
ALLOWED = {"src/sanity_check.py"}


def _guarded_files():
    files = glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
    files += glob.glob(os.path.join(ROOT, "tools", "**", "*.py"), recursive=True)
    files.append(os.path.join(ROOT, "run_pipeline.py"))
    return files


def test_no_ground_truth_references():
    leaks = []
    for path in _guarded_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel in ALLOWED:
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        hits = [tok for tok in FORBIDDEN if tok in text]
        if hits:
            leaks.append(f"{rel}: {hits}")
    assert not leaks, "ground-truth references leaked into: " + "; ".join(leaks)


def test_run_pipeline_only_reaches_gt_through_sanity_flag():
    with open(os.path.join(ROOT, "run_pipeline.py"), encoding="utf-8") as f:
        text = f.read()
    assert text.count("sanity_check import") == 1
    assert "if args.sanity_check:" in text

