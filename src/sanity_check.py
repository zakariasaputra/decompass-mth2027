"""Report-only verification check against the public 5-minute ground truth.

This is the only module in the pipeline allowed to read evaluation/gt/*. It verifies
frame alignment, format integrity, and label polarity on the public check window without
feeding any signal back into model weights or decoder parameters (`RULES.md`). See
`reports/public_sanity_check.md` for the verification table of the submitted output.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", f"{GAME}_possession_gt_public.csv")


def run_sanity_check(csv_path, gt_path=PUBLIC_GT):
    pred = pm.label_codes(pm.read_label_csv(csv_path))
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    blocks = half_video_blocks(GAME)

    graded = np.zeros(len(pred), dtype=bool)
    for _, s, last in blocks:
        graded[s:last + 1] = True

    print("\n" + "=" * 80)
    print("  PUBLIC-WINDOW SANITY CHECK (report only -- no changes are made in response)")
    print("=" * 80)
    checks = {
        "row count matches the video": len(pred) == len(gt),
        "X exactly on ungraded frames": bool(np.all((pred == pm.X) == ~graded)),
        "no X inside graded play": bool(np.all(pred[graded] != pm.X)),
    }
    res = pm.evaluate(gt, pred)
    gt_live = (gt == pm.H) | (gt == pm.A)
    both_live = gt_live & ((pred == pm.H) | (pred == pm.A))
    swapped_acc = float(np.mean(pred[both_live] != gt[both_live])) if both_live.any() else 0.0
    checks["H/A not swapped (team agreement > 50%)"] = swapped_acc < 0.5
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print(f"\n  spellF1 (mean 0.3..0.7) {res['spell_f1_mean']:.4f}   macroF1 {res['macro_f1']:.4f}")
    print(f"  per class F1: H {res['per_class']['H']['f1']:.4f}  A {res['per_class']['A']['f1']:.4f}  "
          f"D {res['per_class']['D']['f1']:.4f}")
    print(f"  spellF1 by IoU: " + "  ".join(f"@{t} {res['spell_f1'][t]['f1']:.4f}" for t in sorted(res['spell_f1'])))
    print(f"  transitions @25f F1 {res['transitions'][25]['f1']:.4f}   predicted spells {res['n_pred_spells']}")
    return checks, res


if __name__ == "__main__":
    run_sanity_check(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "possession_pipeline.csv"))

