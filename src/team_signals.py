"""Per-frame team-level kinematic signals and dead-ball physics on Metrica pitch coordinates."""

import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d

FPS = 25.0
HALF_L, HALF_W = 52.65, 34.0
MAX_PLAYER_SPEED = 12.0  # m/s: bigger jumps are slot switches, not locomotion
GOAL_AREA_X, GOAL_AREA_Y = HALF_L - 5.5, 9.16
CORNER_R = 2.5


def _interp_nan(x):
    """Linear interpolation over NaNs (edges held)."""
    x = x.copy()
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return np.nan_to_num(x)
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def gap_info(vis):
    """seconds since last visible frame, seconds until next visible frame, length of current gap."""
    T = len(vis)
    idx = np.arange(T)
    last = np.where(vis, idx, -10**9)
    last = np.maximum.accumulate(last)
    nxt = np.where(vis, idx, 10**9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.clip(idx - last, 0, 10**6) / FPS
    until = np.clip(nxt - idx, 0, 10**6) / FPS
    gap = np.where(vis, 0, np.minimum(since + until, 10**6 / FPS))
    return since, until, gap


def player_speed(P):
    """(T, K) speed per slot, NaN at gaps and slot switches (> 12 m/s)."""
    v = np.full(P.shape[:2], np.nan, np.float32)
    v[1:] = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(-1)) * FPS
    v[v > MAX_PLAYER_SPEED] = np.nan
    return v


def compute(m):
    """Dict of (T,) float32 arrays from match dict m with keys 'ball', 'home', 'away' in centered meters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(m)


def _compute(m):
    ball, home, away = m["ball"], m["home"], m["away"]
    T = len(ball)
    vis = ~np.isnan(ball[:, 0])
    s = {"vis": vis.astype(np.float32)}

    with np.errstate(invalid="ignore", all="ignore"):
        dh = np.sqrt(((home - ball[:, None]) ** 2).sum(-1))
        da = np.sqrt(((away - ball[:, None]) ** 2).sum(-1))
        s["d_home"] = np.nanmin(np.where(np.isnan(dh), np.inf, dh), axis=1)
        s["d_away"] = np.nanmin(np.where(np.isnan(da), np.inf, da), axis=1)
        s["d_home"][~vis] = np.nan
        s["d_away"][~vis] = np.nan
        s["n_home_3m"] = np.where(vis, (dh < 3).sum(1), np.nan)
        s["n_away_3m"] = np.where(vis, (da < 3).sum(1), np.nan)
        s["n_home_10m"] = np.where(vis, (dh < 10).sum(1), np.nan)
        s["n_away_10m"] = np.where(vis, (da < 10).sum(1), np.nan)

        # ball kinematics on a lightly smoothed, gap-interpolated track (only valid where visible)
        bx = uniform_filter1d(_interp_nan(ball[:, 0]), 3)
        by = uniform_filter1d(_interp_nan(ball[:, 1]), 3)
        vx = np.gradient(bx) * FPS
        vy = np.gradient(by) * FPS
        sp = np.sqrt(vx ** 2 + vy ** 2)
        sp[~vis] = np.nan
        s["ball_speed"] = sp
        acc = np.abs(np.gradient(uniform_filter1d(np.nan_to_num(sp), 3))) * FPS
        acc[~vis] = np.nan
        s["ball_acc"] = acc
        # direction change over +-3 frames (cosine)
        k = 3
        v0x, v0y = np.roll(vx, k), np.roll(vy, k)
        v1x, v1y = np.roll(vx, -k), np.roll(vy, -k)
        cos = (v0x * v1x + v0y * v1y) / (np.hypot(v0x, v0y) * np.hypot(v1x, v1y) + 1e-6)
        cos[~vis] = np.nan
        s["ball_turn"] = 1.0 - cos
        s["ball_x"] = np.where(vis, ball[:, 0], np.nan)
        s["ball_absx"] = np.abs(s["ball_x"])
        s["ball_absy"] = np.where(vis, np.abs(ball[:, 1]), np.nan)
        s["out_x"] = s["ball_absx"] - HALF_L
        s["out_y"] = s["ball_absy"] - HALF_W
        s["out"] = np.maximum(s["out_x"], s["out_y"])  # >0 = outside the pitch

        vh = player_speed(home)
        va = player_speed(away)
        allv = np.concatenate([vh, va], 1)
        s["pl_speed_med"] = np.nanmedian(allv, 1)
        s["pl_speed_mean"] = np.nanmean(allv, 1)
        s["pl_frac_walk"] = np.nanmean(np.where(np.isnan(allv), np.nan, allv < 2.0), 1)
        s["pl_frac_run"] = np.nanmean(np.where(np.isnan(allv), np.nan, allv > 4.0), 1)
        s["home_speed"] = np.nanmean(vh, 1)
        s["away_speed"] = np.nanmean(va, 1)
        s["n_home_vis"] = (~np.isnan(home[:, :, 0])).sum(1).astype(np.float32)
        s["n_away_vis"] = (~np.isnan(away[:, :, 0])).sum(1).astype(np.float32)

        # team shape: centroid x, spread
        hx = np.nanmean(home[:, :, 0], 1)
        ax = np.nanmean(away[:, :, 0], 1)
        s["centroid_dx"] = hx - ax
        s["home_spread"] = np.nanstd(home[:, :, 0], 1) + np.nanstd(home[:, :, 1], 1)
        s["away_spread"] = np.nanstd(away[:, :, 0], 1) + np.nanstd(away[:, :, 1], 1)
        # ball position relative to team centroids
        s["ball_minus_hx"] = s["ball_x"] - hx
        s["ball_minus_ax"] = s["ball_x"] - ax

    since, until, gap = gap_info(vis)
    s["since_vis"], s["until_vis"], s["gap_len"] = since, until, gap
    s.update(physics(s, ball, vis, allv))
    for k_ in list(s):
        s[k_] = np.asarray(s[k_], dtype=np.float32)
    return s


def _run_length(mask):
    """For each frame, length (s) of the consecutive True run it belongs to so far (0 if False)."""
    out = np.zeros(len(mask), np.float32)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        out[i] = run
    return out / FPS


def physics(s, ball, vis, allv):
    f = {}
    bx, by = ball[:, 0], ball[:, 1]
    sp = np.nan_to_num(s["ball_speed"], nan=0.0)
    out = np.nan_to_num(s["out"], nan=-99.0)
    # 1. causal out-of-bounds state
    oob = np.zeros(len(ball), np.float32)
    since_out = np.full(len(ball), 120.0, np.float32)
    state, last = 0, -10**9
    for i in range(len(ball)):
        if vis[i] and out[i] > 0.5:
            state, last = 1, i
        elif state and vis[i] and out[i] < -0.5 and sp[i] > 3.0:
            state = 0
        oob[i] = state
        since_out[i] = min((i - last) / FPS, 120.0)
    f["oob_state"] = oob
    f["since_out"] = since_out
    # 2. stationary ball and where it rests
    still = vis & (sp < 0.7)
    f["ball_still_run"] = _run_length(still)
    with np.errstate(invalid="ignore"):
        dline = np.minimum(HALF_L - np.abs(bx), HALF_W - np.abs(by))
        f["dist_line"] = np.where(vis, dline, np.nan)
        f["in_goal_area"] = np.where(vis, (np.abs(bx) > GOAL_AREA_X) & (np.abs(by) < GOAL_AREA_Y), np.nan)
        f["at_corner"] = np.where(vis, np.hypot(HALF_L - np.abs(bx), HALF_W - np.abs(by)) < CORNER_R, np.nan)
        f["at_centre_spot"] = np.where(vis, np.hypot(bx, by) < 1.5, np.nan)
        f["still_at_line"] = (still & (np.nan_to_num(dline, nan=99) < 2.0)).astype(np.float32)
        # 3. collective slowdown
        f["pl_frac_still"] = np.nanmean(np.where(np.isnan(allv), np.nan, allv < 1.0), 1)
    f["slow_run"] = _run_length(np.nan_to_num(s["pl_speed_med"], nan=0.0) < 1.5)
    return f
