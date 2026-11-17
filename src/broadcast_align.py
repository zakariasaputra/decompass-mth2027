"""Algorithmic cross-camera alignment and ball/player fusion (Broadcast -> Tactical).

The broadcast recording (`164,488` frames) is unsynchronized to the tactical recording
(`199,319` frames) and contains cuts, replays, and trimmed halftime/pre-match intervals.
However, during each half of live play, both cameras record the same real-time match at 25 fps.

This module:
1. Algorithmically discovers the time-varying frame offset `offset(t) = t_tactical - t_broadcast`
   for each half via chunked cross-correlation and minimum ball-distance refinement.
2. Filters out broadcast cuts, close-ups, and replays using two geometric invariants:
   - Broadcast homography validity (`valid == True` and a reprojection-error ceiling).
   - Player-constellation matching: the median distance from visible broadcast players to the
     nearest same-team tactical player must be small.
3. Recovers missing tactical ball positions from the live broadcast camera when the broadcast ball
   is spatially plausible relative to the surrounding tactical trajectory.

The three thresholds are ``bcast.*`` parameters. When they are ``None`` (the calibrated
configuration) they are derived at run time from UNLABELLED distributions of the provided data:
an Otsu split for the bimodal constellation distance, and percentiles / an Otsu split for the
reprojection error and ball plausibility. No labels are involved.
"""

import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, half_video_blocks  # noqa: E402
from src.cache_atd import build_cache  # noqa: E402
from starter.load_homography import load as load_homography  # noqa: E402

# Bump whenever the alignment/threshold logic changes, so cached results are rebuilt.
ALIGN_VERSION = 2


def aligned_cache_path(game=GAME, data_dir=None):
    root = data_dir or DATA_DIR
    return os.path.join(root, "cache", f"{game}_broadcast_aligned.npz")


def otsu_threshold(values, n_bins=256, log=False):
    """Otsu's between-class-variance threshold of a 1-D sample (optionally in log space)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if log:
        v = np.log(np.maximum(v, 1e-3))
    hist, edges = np.histogram(v, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    thr = centers[int(np.argmax(between))]
    return float(np.exp(thr)) if log else float(thr)


def find_half_offset_curve(t_data, b_data, b_valid, t_start, t_end, n_chunks=6):
    """Algorithmically discover the frame offset `t - b` across a half."""
    length, width = t_data["pitch"]
    t_bx = np.nan_to_num(t_data["ball_x"][t_start:t_end] - 0.5)
    b_bx = np.nan_to_num(b_data["ball_x"] - 0.5)

    # Coarse global cross-correlation over the half
    corr = np.correlate(b_bx, t_bx, mode="valid")
    coarse_offset = int(t_start - int(np.argmax(corr)))

    def _median_ball_dist(t_frames, cand, min_count):
        bi = t_frames - cand
        bic = np.clip(bi, 0, len(b_valid) - 1)
        valid = (
            (bi >= 0)
            & (bi < len(b_data["ball_x"]))
            & b_valid[bic]
            & (~np.isnan(t_data["ball_x"][t_frames]))
            & (~np.isnan(b_data["ball_x"][bic]))
        )
        if valid.sum() <= min_count:
            return np.inf
        ti_v, bi_v = t_frames[valid], bi[valid]
        dist = np.hypot(
            (t_data["ball_x"][ti_v] - b_data["ball_x"][bi_v]) * length,
            (t_data["ball_y"][ti_v] - b_data["ball_y"][bi_v]) * width,
        )
        return float(np.median(dist))

    # Refine the coarse offset within +/- 30 frames using the median ball distance
    t_all = np.arange(t_start, t_end + 1)
    best_global_off, best_global_med = coarse_offset, np.inf
    for delta in range(-30, 31):
        med = _median_ball_dist(t_all, coarse_offset + delta, 500)
        if med < best_global_med:
            best_global_med, best_global_off = med, coarse_offset + delta

    # Local chunk refinement (handles small +/- 1 frame clock drift over 45 minutes)
    chunk_edges = np.linspace(t_start, t_end + 1, n_chunks + 1, dtype=int)
    chunk_centers, chunk_offsets = [], []
    for i in range(n_chunks):
        t_chunk = np.arange(chunk_edges[i], chunk_edges[i + 1])
        best_c_off, best_c_med = best_global_off, np.inf
        for delta in range(-5, 6):
            med = _median_ball_dist(t_chunk, best_global_off + delta, 100)
            if med < best_c_med:
                best_c_med, best_c_off = med, best_global_off + delta
        chunk_centers.append(0.5 * (chunk_edges[i] + chunk_edges[i + 1]))
        chunk_offsets.append(best_c_off)

    return np.round(np.interp(t_all, chunk_centers, chunk_offsets)).astype(np.int32)


def _bcast_param_hash(P):
    keys = ("bcast.max_ball_disc", "bcast.max_player_dist", "bcast.max_reproj")
    items = sorted((k, P.get(k)) for k in keys)
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build_broadcast_alignment(P=None, game=GAME, data_dir=None, force=False):
    """Align broadcast ATD onto the tactical timeline and cache the fused signals."""
    if P is None:
        P = {"bcast.max_ball_disc": None, "bcast.max_player_dist": None, "bcast.max_reproj": None}

    out_path = aligned_cache_path(game, data_dir)
    key = f"v{ALIGN_VERSION}-" + _bcast_param_hash(P)
    if os.path.isfile(out_path) and not force:
        t0 = time.time()
        res = dict(np.load(out_path, allow_pickle=False))
        if str(res.get("params_key", "")) == key:
            print(f"[broadcast_align] Loaded aligned broadcast signals ({key}) in {time.time() - t0:.2f}s")
            return res

    t0 = time.time()
    t_data = build_cache(game=game, data_dir=data_dir, source="tactical")
    b_data = build_cache(game=game, data_dir=data_dir, source="broadcast")
    _, b_reproj, b_valid = load_homography(game=game, data_dir=data_dir, source="broadcast")

    n_frames = len(t_data["ball_x"])
    length, width = t_data["pitch"]

    b_frame_for_t = np.full(n_frames, -1, dtype=np.int32)
    med_player_dist = np.full(n_frames, np.nan, dtype=np.float32)

    t_team = t_data["team_of_slot"]
    b_team = b_data["team_of_slot"]
    same_team_mask = (b_team[:, None] == t_team[None, :])[None, :, :]

    # 1. Offsets + constellation distance for every tactical frame of each half
    for _, t_start, t_end in half_video_blocks(game):
        offsets = find_half_offset_curve(t_data, b_data, b_valid, t_start, t_end)
        t_idx = np.arange(t_start, t_end + 1)
        bi = t_idx - offsets
        in_bounds = (bi >= 0) & (bi < len(b_data["ball_x"]))
        t_idx, bi = t_idx[in_bounds], bi[in_bounds]
        b_frame_for_t[t_idx] = bi

        for s in range(0, len(t_idx), 4000):
            e = min(len(t_idx), s + 4000)
            ti_c, bi_c = t_idx[s:e], bi[s:e]
            dx = (b_data["player_x"][bi_c, :, None] - t_data["player_x"][ti_c, None, :]) * length
            dy = (b_data["player_y"][bi_c, :, None] - t_data["player_y"][ti_c, None, :]) * width
            d2 = dx * dx + dy * dy
            d2 = np.where(same_team_mask & (~np.isnan(d2)), d2, np.inf)
            min_d = np.sqrt(np.min(d2, axis=2))
            min_d = np.where(np.isinf(min_d), np.nan, min_d)
            with np.errstate(all="ignore"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med_player_dist[ti_c] = np.nanmedian(min_d, axis=1)

    mapped = b_frame_for_t >= 0
    bmap = np.clip(b_frame_for_t, 0, len(b_valid) - 1)
    hom_ok = mapped & b_valid[bmap]

    # 2. Thresholds: explicit values, or derived from unlabelled distributions
    thr_player = P["bcast.max_player_dist"]
    if thr_player is None:
        sample = med_player_dist[hom_ok & np.isfinite(med_player_dist)]
        thr_player = otsu_threshold(np.clip(sample, 0.05, 30.0), log=True)
    thr_reproj = P["bcast.max_reproj"]
    if thr_reproj is None:
        # Tukey far-outlier fence (Q3 + 3 IQR) of the reprojection error on frames that already pass
        # the constellation test: removes gross homography failures, keeps ordinary live frames.
        sample = b_reproj[bmap][hom_ok & (med_player_dist < thr_player)]
        q1, q3 = np.nanpercentile(sample, [25, 75])
        thr_reproj = float(q3 + 3.0 * (q3 - q1))

    b_live = hom_ok & (b_reproj[bmap] < thr_reproj) & np.isfinite(med_player_dist) & (med_player_dist < thr_player)
    ti_live = np.flatnonzero(b_live)
    bi_live = b_frame_for_t[ti_live]

    b_ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    b_ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    b_ball_x[ti_live] = b_data["ball_x"][bi_live]
    b_ball_y[ti_live] = b_data["ball_y"][bi_live]

    # 3. Ball recovery: tactical primary; broadcast when tactical is NaN and spatially plausible
    t_vis = ~np.isnan(t_data["ball_x"])
    b_vis = ~np.isnan(b_ball_x)
    valid_t = np.flatnonzero(t_vis)
    all_idx = np.arange(n_frames)
    t_bx_interp = np.interp(all_idx, valid_t, t_data["ball_x"][valid_t])
    t_by_interp = np.interp(all_idx, valid_t, t_data["ball_y"][valid_t])
    b_to_t_interp_m = np.hypot((b_ball_x - t_bx_interp) * length, (b_ball_y - t_by_interp) * width)

    thr_ball = P["bcast.max_ball_disc"]
    if thr_ball is None:
        # Candidate recoveries are a mixture of correct broadcast detections (close to the tactical
        # trajectory) and false ball hypotheses (far away). Split the mixture with Otsu in log space.
        cand = b_to_t_interp_m[(~t_vis) & b_vis]
        thr_ball = otsu_threshold(np.clip(cand, 0.05, 150.0), log=True)

    b_plausible = b_vis & (b_to_t_interp_m < thr_ball)
    fused_ball_x = np.where(t_vis, t_data["ball_x"], np.where(b_plausible, b_ball_x, np.nan)).astype(np.float32)
    fused_ball_y = np.where(t_vis, t_data["ball_y"], np.where(b_plausible, b_ball_y, np.nan)).astype(np.float32)
    # Source flag: 0 = missing, 1 = tactical, 2 = broadcast recovery, 3 = both visible
    ball_source = np.zeros(n_frames, dtype=np.int8)
    ball_source[t_vis & (~b_vis)] = 1
    ball_source[(~t_vis) & b_plausible] = 2
    ball_source[t_vis & b_vis] = 3

    thresholds = np.array([thr_reproj, thr_player, thr_ball], dtype=np.float64)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        params_key=np.array(key),
        thresholds=thresholds,
        b_frame_for_t=b_frame_for_t,
        med_player_dist=med_player_dist,
        b_to_t_interp_m=b_to_t_interp_m.astype(np.float32),
        b_live=b_live,
        b_ball_x=b_ball_x,
        b_ball_y=b_ball_y,
        fused_ball_x=fused_ball_x,
        fused_ball_y=fused_ball_y,
        ball_source=ball_source,
    )
    print(
        f"[broadcast_align] Aligned in {time.time() - t0:.2f}s | thresholds: reproj<{thr_reproj:.2f}, "
        f"constellation<{thr_player:.2f} m, ball<{thr_ball:.2f} m | live frames: {int(b_live.sum())} | "
        f"recovered balls: {int(np.sum(ball_source == 2))}"
    )
    return dict(np.load(out_path, allow_pickle=False))


if __name__ == "__main__":
    build_broadcast_alignment(force=True)
