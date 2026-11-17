"""Single-command, deterministic reproduction of the possession submission.

    python run_pipeline.py                                   # -> evaluation/results/possession_pipeline.csv
    python run_pipeline.py --out submission/Decompass.csv    # -> official submission artifact
    python run_pipeline.py --rebuild-cache                   # rebuild every cache from raw ./data
    python run_pipeline.py --sanity-check                    # + report-only public-window check

Stages:
  1. Binary ATD caches (src/cache_atd.py)
  2. Broadcast-to-tactical alignment and ball recovery (src/broadcast_align.py)
  3. Ball-free Nova PathCRF inference (src/models/nova_pathcrf.py)
  4. Team signals (src/team_signals.py), 3-model ensemble blending, pre-kickoff gate
     (src/features.py:detect_kickoff_frame), and 3-state Viterbi (src/models/super_hybrid.py)
  5. Format validation with evaluation/validate_submission.py rules

Models and weights are trained or calibrated on external matches only. Nothing here reads
the provided match's ground truth unless --sanity-check is passed.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import evaluation.possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from src.broadcast_align import build_broadcast_alignment  # noqa: E402
from src.cache_atd import build_cache  # noqa: E402
from src.models.nova_pathcrf import build_nova_pathcrf_signals  # noqa: E402
from src.models.super_hybrid import load_ensemble_config, predict_super_hybrid  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

CODE_TO_STR = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}
DEFAULT_OUT = os.path.join(HERE, "evaluation", "results", "possession_pipeline.csv")


def summarize(labels, blocks):
    """Ground-truth-free summary of a label sequence (spells, mean duration, shares)."""
    runs = [(b[1], b[2] + 1) for b in blocks]
    stats = pm.possession_stats(labels, runs)
    graded = np.zeros(len(labels), dtype=bool)
    for s, e in runs:
        graded[s:e] = True
    return {
        "n_spells": int(stats["n_spells"]),
        "mean_duration_s": float(stats["mean_duration_s"]),
        "home_share_live": float(stats["home_share"]),
        "dead_share_graded": float(np.mean(labels[graded] == pm.D)),
    }


def write_submission_csv(out_csv, pred_codes):
    """UTF-8 CSV with header `frame,label` and rows 0..N-1 (LF line endings, deterministic)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    lines = ["frame,label\n"] + [f"{i},{CODE_TO_STR[int(c)]}\n" for i, c in enumerate(pred_codes)]
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(out_csv=DEFAULT_OUT, game=GAME, rebuild_cache=False):
    t_start = time.time()
    blocks = half_video_blocks(game)

    print("=" * 80)
    print(f"  POSSESSION PIPELINE ({game})")
    print("=" * 80)

    t0 = time.time()
    t_data = build_cache(game=game, source="tactical", force=rebuild_cache)
    build_cache(game=game, source="broadcast", force=rebuild_cache)
    print(f"[1] ATD caches ready ({time.time() - t0:.2f}s)")

    events = load_smart_tagging(game=game)

    t0 = time.time()
    b_align = build_broadcast_alignment(game=game, force=rebuild_cache)
    graded = np.zeros(len(t_data["ball_x"]), dtype=bool)
    for _, start, last in blocks:
        graded[start : last + 1] = True
    t_vis = ~np.isnan(t_data["ball_x"])
    f_vis = b_align["ball_source"] > 0
    n_rec = int((b_align["ball_source"][graded] == 2).sum())
    print(
        f"[2] Broadcast fusion ready: ball visible on {100 * f_vis[graded].mean():.1f}% of graded frames "
        f"(up from {100 * t_vis[graded].mean():.1f}% tactical; {n_rec:,} recovered) ({time.time() - t0:.2f}s)"
    )

    ens_cfg = load_ensemble_config()
    t0 = time.time()
    pc_probs = build_nova_pathcrf_signals(
        t_data, blocks, game=game, stride_seconds=float(ens_cfg["pathcrf_stride_seconds"]), force=rebuild_cache
    )
    print(f"[3] Nova PathCRF ball-free signals ready ({time.time() - t0:.2f}s)")

    t0 = time.time()
    hybrid_res = predict_super_hybrid(
        t_data,
        b_align["fused_ball_x"],
        b_align["fused_ball_y"],
        blocks,
        events=events,
        pathcrf_probs=pc_probs,
        ensemble_cfg=ens_cfg,
    )
    pred = hybrid_res["labels"]
    s = summarize(pred, blocks)
    print(
        f"[4] 3-model ensemble decoded: {s['n_spells']} live spells, mean {s['mean_duration_s']:.1f}s, "
        f"dead {100 * s['dead_share_graded']:.1f}% of graded, "
        f"home share of live {100 * s['home_share_live']:.1f}% ({time.time() - t0:.2f}s)"
    )

    write_submission_csv(out_csv, pred)
    reason = pm.validate_submission(pm.read_label_csv(out_csv), GAMES[game]["n_video_frames"])
    if reason:
        raise RuntimeError(f"Submission validation failed: {reason}")
    counts = {name: int(np.sum(pred == idx)) for idx, name in enumerate(pm.LABELS)}
    print(f"[5] Wrote {out_csv}")
    print(f"    format validation PASSED | counts {counts}")
    print(f"    sha256 {sha256_file(out_csv)}")
    print(f"Total runtime {time.time() - t_start:.2f}s")
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--rebuild-cache", action="store_true", help="rebuild every cache from raw ./data")
    ap.add_argument(
        "--sanity-check",
        action="store_true",
        help="after writing, run the report-only public-window check",
    )
    args = ap.parse_args()
    run_pipeline(out_csv=args.out, rebuild_cache=args.rebuild_cache)
    if args.sanity_check:
        from src.sanity_check import run_sanity_check

        run_sanity_check(args.out)


if __name__ == "__main__":
    main()
