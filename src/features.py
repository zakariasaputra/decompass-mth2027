"""Spatiotemporal, kinematic, and tactical feature extraction.

Two layers:

1. ``compute_core_features`` -- match-agnostic. Takes player/ball positions (normalised 0..1), team
   assignment, graded blocks and pitch size, and returns ball kinematics, player-ball proximity and
   collective player tempo. The SAME function runs on the provided match and on the external
   calibration matches, so parameters calibrated externally mean the same thing here.
2. ``extract_features`` -- provided match (BAR-ATH). Adds broadcast-camera ball fusion, the
   second-half pre-kickoff mask (FAQ.md), and Smart Tagging set-piece windows, then caches to
   ``data/cache/BAR-ATH_features.npz`` (keyed by a hash of the parameters it depends on).
"""

import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.ndimage import convolve1d, uniform_filter1d
from scipy.signal import savgol_coeffs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, half_video_blocks  # noqa: E402

# Set-piece type integer codes
SP_NONE = 0
SP_KICKOFF_START = 1
SP_KICKOFF_GOAL = 2
SP_GOAL_KICK = 3
SP_CORNER = 4
SP_FREE_KICK = 5
SP_THROW_IN = 6

SP_TYPE_MAP = {
    "kick off (start)": SP_KICKOFF_START,
    "kick off (after goal)": SP_KICKOFF_GOAL,
    "goal kick": SP_GOAL_KICK,
    "corner kick": SP_CORNER,
    "free kick": SP_FREE_KICK,
    "throw-in": SP_THROW_IN,
}

# Smart Tagging phase weights. Only used when hmm.st_w_* / dead.buildup_requires_phase are non-zero
# (they are zero in the calibrated configuration: phase priors cannot be calibrated externally).
OFFENSIVE_CODES = {
    "BUILD UP": 1.2,
    "PROGRESSION": 1.2,
    "CREATING CHANCES": 1.2,
    "ATTACKING TRANSITION": 1.2,
    "LONG BALL": 0.9,
    "BALL IN FINAL THIRD": 0.8,
    "BALL IN THE BOX": 0.8,
    "PLAYERS IN THE BOX": 0.6,
}

DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD": 1.2,
    "DEFENDING IN MIDDLE THIRD": 1.2,
    "DEFENDING IN DEFENSIVE THIRD": 1.2,
    "DEFENSIVE TRANSITION": 1.2,
}

FEATURE_PARAM_PREFIXES = ("feat.", "bcast.")
# Bump whenever feature logic (or the broadcast alignment it consumes) changes.
FEATURE_VERSION = 5


def features_cache_path(game=GAME, data_dir=None):
    root = data_dir or DATA_DIR
    return os.path.join(root, "cache", f"{game}_features.npz")


def _compute_gaps_and_interpolation(bx, by, blocks, length, width, fps, max_interp_gap, max_speed_mps):
    """Compute missing-ball gap lengths and linearly interpolate plausible short in-flight gaps."""
    n = len(bx)
    gap_len = np.zeros(n, dtype=np.int32)
    interp_x = bx.copy()
    interp_y = by.copy()
    is_interp = np.zeros(n, dtype=bool)

    vis = ~np.isnan(bx)
    for _, start, last in blocks:
        stop = last + 1
        missing = (~vis[start:stop]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        g_starts = np.flatnonzero(edges == 1) + start
        g_stops = np.flatnonzero(edges == -1) + start

        for gs, ge in zip(g_starts, g_stops):
            glen = ge - gs
            gap_len[gs:ge] = glen
            if gs > start and ge < stop and glen <= max_interp_gap:
                x0, y0 = bx[gs - 1], by[gs - 1]
                x1, y1 = bx[ge], by[ge]
                dist_m = np.hypot((x1 - x0) * length, (y1 - y0) * width)
                req_speed = dist_m / (glen / fps)
                if req_speed <= max_speed_mps:
                    alpha = np.linspace(0.0, 1.0, glen + 2, dtype=np.float32)[1:-1]
                    interp_x[gs:ge] = x0 + alpha * (x1 - x0)
                    interp_y[gs:ge] = y0 + alpha * (y1 - y0)
                    is_interp[gs:ge] = True

    return gap_len, interp_x, interp_y, is_interp


def _nan_smooth1d(arr, window):
    """Fast NaN-aware uniform moving average of width `window`."""
    valid = (~np.isnan(arr)).astype(np.float32)
    filled = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    num = uniform_filter1d(filled, size=window, mode="nearest")
    den = uniform_filter1d(valid, size=window, mode="nearest")
    return np.where(den > 1e-5, num / np.maximum(den, 1e-5), 0.0).astype(np.float32)


def player_speed_from_positions(px_m, py_m, fps=25.0):
    """Per-player speed (m/s) from 5-frame central differences of positions (NaN where untracked or slot switch)."""
    n = px_m.shape[0]
    speed = np.full(px_m.shape, np.nan, dtype=np.float32)
    if n > 4:
        dx = px_m[4:] - px_m[:-4]
        dy = py_m[4:] - py_m[:-4]
        speed[2:-2] = (np.hypot(dx, dy) * (fps / 4.0)).astype(np.float32)
    speed[speed > 12.0] = np.nan
    return speed


def deduplicate_slots(px_m, py_m, team, dist_thr=0.35):
    """Mask out redundant player slots when two tracks of the same team are coincident within dist_thr (m)."""
    if dist_thr <= 0:
        return px_m, py_m
    px_clean = px_m.copy()
    py_clean = py_m.copy()
    team = np.asarray(team)
    dist_sq_thr = dist_thr * dist_thr
    for t_val in (0, 1):
        idx = np.flatnonzero(team == t_val)
        if len(idx) < 2:
            continue
        sub_x = px_clean[:, idx]
        sub_y = py_clean[:, idx]
        for i in range(len(idx)):
            xi, yi = sub_x[:, i], sub_y[:, i]
            vi = ~np.isnan(xi)
            if not np.any(vi):
                continue
            for j in range(i + 1, len(idx)):
                xj, yj = sub_x[:, j], sub_y[:, j]
                v_both = vi & (~np.isnan(xj))
                if not np.any(v_both):
                    continue
                dx = xi[v_both] - xj[v_both]
                close_x = np.abs(dx) < dist_thr
                if not np.any(close_x):
                    continue
                dy = yi[v_both][close_x] - yj[v_both][close_x]
                d2 = dx[close_x] ** 2 + dy ** 2
                coinc_sub = np.flatnonzero(d2 < dist_sq_thr)
                if len(coinc_sub) > 0:
                    coinc_idx = np.flatnonzero(v_both)[close_x][coinc_sub]
                    px_clean[coinc_idx, idx[j]] = np.nan
                    py_clean[coinc_idx, idx[j]] = np.nan
    return px_clean, py_clean


def player_speed_savgol(px_m, py_m, fps=25.0, window=7, polyorder=2):
    """Vectorized per-player speed (m/s) using Savitzky-Golay FIR differentiation."""
    if window < 3 or window % 2 == 0 or polyorder >= window:
        return player_speed_from_positions(px_m, py_m, fps)
    coeffs = savgol_coeffs(window, polyorder, deriv=1, delta=1.0 / fps)
    nan_mask = np.isnan(px_m) | np.isnan(py_m)
    invalid = uniform_filter1d(nan_mask.astype(np.float32), size=window, axis=0) > 1e-5
    vx = convolve1d(np.nan_to_num(px_m, nan=0.0), weights=coeffs, axis=0, mode="constant", cval=0.0)
    vy = convolve1d(np.nan_to_num(py_m, nan=0.0), weights=coeffs, axis=0, mode="constant", cval=0.0)
    speed = np.hypot(vx, vy).astype(np.float32)
    speed[invalid] = np.nan
    speed[speed > 12.0] = np.nan
    return speed


def compute_ppcf_at_ball(px_m, py_m, ps, team, bx_m, by_m, reaction_time=0.7, max_speed=8.0, lambda_rate=4.3):
    """Spearman's Potential Pitch Control Field (PPCF) evaluated at the ball location."""
    n_frames = len(bx_m)
    ppcf_h = np.zeros(n_frames, dtype=np.float32)
    ppcf_a = np.zeros(n_frames, dtype=np.float32)
    team = np.asarray(team)
    home_mask = (team == 0)
    away_mask = (team == 1)
    reach_radius = 1.0
    dx = px_m - bx_m[:, None]
    dy = py_m - by_m[:, None]
    dist = np.hypot(dx, dy)
    eff_speed = np.clip(np.nan_to_num(ps, nan=1.5), 1.5, max_speed)
    t_intercept = reaction_time + np.maximum(0.0, dist - reach_radius) / eff_speed
    t_intercept[np.isnan(dist)] = np.inf
    min_t = np.nanmin(t_intercept, axis=1, keepdims=True)
    min_t = np.where(np.isfinite(min_t), min_t, 0.0)
    norm_t = np.clip(t_intercept - min_t, 0.0, 5.0)
    exp_intensity = np.where(np.isfinite(t_intercept), np.exp(-lambda_rate * norm_t), 0.0)
    int_home = np.sum(exp_intensity[:, home_mask], axis=1)
    int_away = np.sum(exp_intensity[:, away_mask], axis=1)
    total_int = int_home + int_away
    valid = total_int > 1e-6
    ppcf_h[valid] = (int_home[valid] / total_int[valid]).astype(np.float32)
    ppcf_a[valid] = (int_away[valid] / total_int[valid]).astype(np.float32)
    return ppcf_h, ppcf_a


def compute_residency(d_home, d_away, tau_frames=12.0, ctrl_radius=2.0):
    """Continuous possession residency: exponential memory of nearest-player ball dominance."""
    n_frames = len(d_home)
    decay = np.exp(-1.0 / max(1.0, float(tau_frames)))
    h_active = (d_home < d_away) & (d_home <= ctrl_radius)
    a_active = (d_away < d_home) & (d_away <= ctrl_radius)
    cur_h, cur_a = 0.0, 0.0
    res_h = np.zeros(n_frames, dtype=np.float32)
    res_a = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        cur_h = cur_h * decay + (1.0 if h_active[i] else 0.0)
        cur_a = cur_a * decay + (1.0 if a_active[i] else 0.0)
        res_h[i] = cur_h
        res_a[i] = cur_a
    # Map to [0, 1] range: (1 - exp(-res / tau))
    sat_h = 1.0 - np.exp(-res_h / max(1.0, float(tau_frames)))
    sat_a = 1.0 - np.exp(-res_a / max(1.0, float(tau_frames)))
    return sat_h.astype(np.float32), sat_a.astype(np.float32)


def detect_kickoff_frame(ball_x, ball_y, blocks, pitch, events=None, fps=25.0):
    """Pre-kickoff mask per half: uses Smart Tagging 'Kick Off (Start)' window + minimum ball-to-center distance when events are provided, falling back to ball within 3.5 m of center in the first 120 s when events=None."""
    n_frames = len(ball_x)
    pre_kickoff = np.zeros(n_frames, dtype=bool)
    length, width = float(pitch[0]), float(pitch[1])
    cx, cy = length / 2.0, width / 2.0
    bx_m = np.asarray(ball_x) * length
    by_m = np.asarray(ball_y) * width
    d_center = np.hypot(bx_m - cx, by_m - cy)

    ko_events = {}
    if events:
        for ev in events:
            if ev.get("code") == "SET PIECES" and "kick off (start)" in ev.get("tags", {}).get("Type", "").lower():
                half_str = ev.get("tags", {}).get("Half", "")
                h_num = 1 if "1st" in half_str else (2 if "2nd" in half_str else 0)
                ko_events[h_num] = (ev["start_frame"], ev["end_frame"])

    for h, start, last in blocks:
        ko_frame = None
        if h in ko_events:
            ev_s, ev_e = ko_events[h]
            s = max(start, ev_s)
            e = min(last + 1, ev_e)
            if e > s:
                sub_d = d_center[s:e]
                if np.any(~np.isnan(sub_d)):
                    min_offset = int(np.nanargmin(sub_d))
                    ko_frame = s + min_offset + 1
        if ko_frame is None:
            search_end = min(last + 1, start + int(120 * fps))
            sub_d = d_center[start:search_end]
            valid_center = np.flatnonzero((~np.isnan(sub_d)) & (sub_d < 3.5))
            if len(valid_center) > 0:
                min_offset = int(np.nanargmin(sub_d[valid_center]))
                ko_frame = start + valid_center[min_offset] + 1

        if ko_frame is not None and ko_frame > start:
            pre_kickoff[start:ko_frame] = True
    return pre_kickoff


def compute_core_features(player_x, player_y, team, ball_x, ball_y, blocks, pitch, P,
                          player_s=None, ball_source=None, fps=25.0):
    """Match-agnostic features. Positions are normalised 0..1; ``team`` is 0=home, 1=away, -1=other."""
    n_frames = len(ball_x)
    length, width = float(pitch[0]), float(pitch[1])

    # 1. Timeline & grading masks
    graded_mask = np.zeros(n_frames, dtype=bool)
    half_id = np.zeros(n_frames, dtype=np.int8)
    for h, start, last in blocks:
        graded_mask[start : last + 1] = True
        half_id[start : last + 1] = h

    # 2. Ball trajectory & kinematics
    ball_x = np.asarray(ball_x, dtype=np.float32)
    ball_y = np.asarray(ball_y, dtype=np.float32)
    ball_vis = ~np.isnan(ball_x)
    if ball_source is None:
        ball_source = ball_vis.astype(np.int8)

    gap_len, interp_x, interp_y, is_interp = _compute_gaps_and_interpolation(
        ball_x, ball_y, blocks, length, width, fps,
        int(P["feat.interp_max_gap"]), float(P["feat.interp_max_speed"]),
    )

    # Ball velocity & acceleration (5-frame / 0.2 s central differences on the interpolated path)
    bx_m = interp_x * length
    by_m = interp_y * width
    vx_raw = np.zeros(n_frames, dtype=np.float32)
    vy_raw = np.zeros(n_frames, dtype=np.float32)
    vx_raw[2:-2] = (bx_m[4:] - bx_m[:-4]) * (fps / 4.0)
    vy_raw[2:-2] = (by_m[4:] - by_m[:-4]) * (fps / 4.0)
    ball_speed = np.hypot(vx_raw, vy_raw).astype(np.float32)

    ax_raw = np.zeros(n_frames, dtype=np.float32)
    ay_raw = np.zeros(n_frames, dtype=np.float32)
    ax_raw[1:-1] = (vx_raw[2:] - vx_raw[:-2]) * (fps / 2.0)
    ay_raw[1:-1] = (vy_raw[2:] - vy_raw[:-2]) * (fps / 2.0)
    ball_acc = np.hypot(ax_raw, ay_raw).astype(np.float32)

    # Pitch boundary geometry (on observed ball)
    obs_x_m = ball_x * length
    obs_y_m = ball_y * width
    ball_edge_dist = np.minimum(
        np.minimum(obs_x_m, length - obs_x_m), np.minimum(obs_y_m, width - obs_y_m)
    ).astype(np.float32)
    ball_oob = ball_vis & (ball_edge_dist < 0.0)

    # 3. Player-to-ball proximity
    px_m = np.asarray(player_x) * length
    py_m = np.asarray(player_y) * width
    team = np.asarray(team)

    # De-duplicate coincident same-team player tracks
    dedup_dist = float(P.get("feat.dedup_dist", 0.35))
    px_m, py_m = deduplicate_slots(px_m, py_m, team, dist_thr=dedup_dist)

    sg_win = int(P.get("feat.sg_window", 7))
    sg_poly = int(P.get("feat.sg_order", 2))
    if P["feat.speed_source"] == "atd":
        if player_s is None:
            raise ValueError("feat.speed_source='atd' requires a player speed channel")
        ps = np.asarray(player_s, dtype=np.float32)
    elif sg_win > 0:
        ps = player_speed_savgol(px_m, py_m, fps, window=sg_win, polyorder=sg_poly)
    else:
        ps = player_speed_from_positions(px_m, py_m, fps)

    # Unsupervised tempo normalisation: rescale this match's player speeds so that the 75th percentile of
    # its smoothed collective tempo (active open play) equals a fixed anchor. Uses no labels.
    tempo_scale = 1.0
    if int(P.get("feat.tempo_norm", 0)):
        outf = (team == 0) | (team == 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p90_raw = _nan_smooth1d(np.nanpercentile(ps[:, outf], 90, axis=1).astype(np.float32), 51)
        ref = float(np.nanpercentile(p90_raw[graded_mask], 75))
        if np.isfinite(ref) and ref > 0:
            tempo_scale = float(P["feat.tempo_anchor"]) / ref
            ps = (ps * tempo_scale).astype(np.float32)

    d_all = np.hypot(px_m - obs_x_m[:, None], py_m - obs_y_m[:, None])
    d_home_mat = np.where((team[None, :] == 0) & (~np.isnan(d_all)), d_all, np.inf)
    d_away_mat = np.where((team[None, :] == 1) & (~np.isnan(d_all)), d_all, np.inf)
    d_home = np.min(d_home_mat, axis=1).astype(np.float32)
    d_away = np.min(d_away_mat, axis=1).astype(np.float32)

    crowd = P["feat.crowd_radius"]
    n_home_3m = np.sum(d_home_mat < crowd, axis=1).astype(np.int8)
    n_away_3m = np.sum(d_away_mat < crowd, axis=1).astype(np.int8)
    min_d = np.minimum(d_home, d_away)
    in_flight = (~ball_vis) | ((ball_speed > P["feat.inflight_speed"]) & (min_d > P["feat.inflight_mind"]))

    # 4. Advanced tactical spatial features: PPCF and Possession Residency
    ppcf_home, ppcf_away = compute_ppcf_at_ball(px_m, py_m, ps, team, bx_m, by_m)
    res_tau = float(P.get("hmm.res_tau", 12.0))
    res_home, res_away = compute_residency(d_home, d_away, tau_frames=res_tau)

    # 5. Collective player dynamics (players only: team 0/1)
    outfield = (team == 0) | (team == 1)
    ps_play = ps[:, outfield]
    n_tracked = np.sum(~np.isnan(px_m[:, outfield]), axis=1).astype(np.int16)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ps_med = np.nanmedian(ps_play, axis=1).astype(np.float32)
            ps_p90 = np.nanpercentile(ps_play, 90, axis=1).astype(np.float32)
        frac_walking_raw = (np.nansum(ps_play < P["feat.walk_speed"], axis=1) / np.maximum(n_tracked, 1)).astype(np.float32)
        frac_sprinting_raw = (np.nansum(ps_play > P["feat.sprint_speed"], axis=1) / np.maximum(n_tracked, 1)).astype(np.float32)

    return {
        "pitch": np.array([length, width], dtype=np.float64),
        "tempo_scale": np.float32(tempo_scale),
        "graded_mask": graded_mask,
        "half_id": half_id,
        "pre_kickoff": np.zeros(n_frames, dtype=bool),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_source": np.asarray(ball_source, dtype=np.int8),
        "ball_vis": ball_vis,
        "gap_len": gap_len,
        "interp_x": interp_x,
        "interp_y": interp_y,
        "is_interp": is_interp,
        "ball_speed": ball_speed,
        "ball_acc": ball_acc,
        "ball_edge_dist": ball_edge_dist,
        "ball_oob": ball_oob,
        "d_home": d_home,
        "d_away": d_away,
        "ppcf_home": ppcf_home,
        "ppcf_away": ppcf_away,
        "res_home": res_home,
        "res_away": res_away,
        "n_home_3m": n_home_3m,
        "n_away_3m": n_away_3m,
        "in_flight": in_flight,
        "n_tracked": n_tracked,
        "ps_med": ps_med,
        "ps_p90": ps_p90,
        "ps_med_s25": _nan_smooth1d(ps_med, 25),
        "ps_p90_s50": _nan_smooth1d(ps_p90, 51),
        "frac_walking": _nan_smooth1d(frac_walking_raw, 51),
        "frac_sprinting": _nan_smooth1d(frac_sprinting_raw, 51),
        # Smart Tagging channels (empty unless the provided-match wrapper fills them)
        "st_home_prior": np.zeros(n_frames, dtype=np.float32),
        "st_away_prior": np.zeros(n_frames, dtype=np.float32),
        "st_set_piece_type": np.zeros(n_frames, dtype=np.int8),
        "st_set_piece_team": np.full(n_frames, -1, dtype=np.int8),
        "st_goal_window": np.zeros(n_frames, dtype=bool),
    }


def add_smart_tagging(feats, events, home_name="Barcelona", away_name="Atletic Club"):
    """Fill the Smart Tagging channels (phase priors, set-piece windows, goal windows) in place."""
    n_frames = len(feats["ball_x"])
    for ev in events:
        code = (ev["code"] or "").strip().upper()
        s = max(0, min(n_frames, ev["start_frame"]))
        e = max(0, min(n_frames, ev["end_frame"]))
        if e <= s:
            continue
        team_str = (ev["tags"].get("Team") or "").strip()
        ev_team = 0 if team_str == home_name else (1 if team_str == away_name else -1)

        if code in OFFENSIVE_CODES and ev_team in (0, 1):
            key = "st_home_prior" if ev_team == 0 else "st_away_prior"
            feats[key][s:e] += OFFENSIVE_CODES[code]
        elif code in DEFENSIVE_CODES and ev_team in (0, 1):
            # A defensive tag on team T means the OPPOSING team has the ball
            key = "st_away_prior" if ev_team == 0 else "st_home_prior"
            feats[key][s:e] += DEFENSIVE_CODES[code]
        elif code == "SET PIECES":
            sp_str = (ev["tags"].get("Type") or "").strip().lower()
            sp_code = SP_TYPE_MAP.get(sp_str, SP_NONE)
            if sp_code != SP_NONE:
                feats["st_set_piece_type"][s:e] = sp_code
                if ev_team in (0, 1):
                    feats["st_set_piece_team"][s:e] = ev_team
        elif code == "GOALS":
            feats["st_goal_window"][s:e] = True
    return feats


_DEFAULT_FEATURE_PARAMS = {
    "feat.speed_source": "positions",
    "feat.interp_max_gap": 100,
    "feat.interp_max_speed": 32.0,
    "feat.walk_speed": 1.8,
    "feat.sprint_speed": 4.0,
    "feat.tempo_norm": 1,
    "feat.tempo_anchor": 4.08,
    "feat.crowd_radius": 3.5,
    "feat.inflight_speed": 9.5,
    "feat.inflight_mind": 1.6,
    "feat.dedup_dist": 0.35,
    "feat.sg_window": 7,
    "feat.sg_order": 2,
    "bcast.max_ball_disc": None,
    "bcast.max_player_dist": None,
    "bcast.max_reproj": None,
}


def _feature_subset_hash(P, prefixes):
    items = sorted((k, P[k]) for k in P if k.startswith(tuple(prefixes)))
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:12]


def extract_features(P=None, game=GAME, data_dir=None, force=False, verbose=True):
    """Build or load the provided match's feature dictionary (cache keyed by parameter hash)."""
    from src.broadcast_align import build_broadcast_alignment
    from src.cache_atd import build_cache
    from starter.load_smart_tagging import load as load_smart_tagging

    if P is None:
        P = dict(_DEFAULT_FEATURE_PARAMS)
    out_path = features_cache_path(game, data_dir)
    key = f"v{FEATURE_VERSION}-" + _feature_subset_hash(P, FEATURE_PARAM_PREFIXES)
    if os.path.isfile(out_path) and not force:
        t0 = time.time()
        cached = dict(np.load(out_path, allow_pickle=False))
        if str(cached.get("params_key", "")) == key:
            if verbose:
                print(f"[features] Loaded cached features ({key}) in {time.time() - t0:.2f}s")
            cached.pop("params_key", None)
            return cached

    t0 = time.time()
    t_data = build_cache(game=game, data_dir=data_dir, source="tactical")
    b_align = build_broadcast_alignment(P, game=game, data_dir=data_dir, force=force)
    events = load_smart_tagging(game=game, data_dir=data_dir)
    blocks = half_video_blocks(game)

    feats = compute_core_features(
        t_data["player_x"], t_data["player_y"], t_data["team_of_slot"],
        b_align["fused_ball_x"], b_align["fused_ball_y"], blocks, t_data["pitch"], P,
        player_s=t_data["player_s"], ball_source=b_align["ball_source"],
    )
    # The fused ball is "visible" whenever either camera contributed it
    feats["ball_vis"] = feats["ball_source"] > 0

    # Pre-kickoff stretch is dead ball by IFAB Law 8 (Smart Tagging kickoff window + center-ball distance, or <3.5m fallback)
    feats["pre_kickoff"] = detect_kickoff_frame(
        feats["interp_x"], feats["interp_y"], blocks, feats["pitch"], events=events
    )
    add_smart_tagging(feats, events)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, params_key=np.array(key), **feats)
    if verbose:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[features] Extracted {len(feats)} features ({size_mb:.1f} MB, key {key}) in {time.time() - t0:.2f}s")
    return feats


if __name__ == "__main__":
    extract_features(force=True)
