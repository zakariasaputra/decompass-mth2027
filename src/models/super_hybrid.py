"""Super-Hybrid Decompass Possession Engine.

Combines:
1. True Per-Object 2D-Conv TCN (Borghesi et al. 2023 / Horton 2020 trick)
2. Asymmetric Past/Future NaN-Aware HGB Trees
3. Ball-Free Nova PathCRF (SetTransformer + SAB + DynamicSparseCRF, SkillCorner fine-tuned)
4. Ball-Visibility-Gated 3-Model Log-Odds Blending (TCN + HGB + Nova PathCRF)
5. Pre-Kickoff Gate on P(live) via Smart Tagging Kick Off (Start) or center-ball fallback
6. Externally Calibrated 3-State Log-Probability Viterbi Decoder (c_team=5.0, c_dead=30.0, temp=1.0)
"""

import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

import src.team_signals as team_signals

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(ROOT, "models")
ENSEMBLE_CFG_PATH = os.path.join(ROOT, "config", "ensemble_weights.json")

_DEFAULT_ENSEMBLE_CFG = {
    "mode": "visibility_gated",
    "pathcrf_readout": "crf_src",
    "pathcrf_switch_delay_s": 1.0,
    "pathcrf_stride_seconds": 10.0,
    "w_home_visible": {"tcn_A": 0.4875, "hgb_A": 0.2625, "pathcrf": 0.25},
    "w_home_occluded": {"tcn_A": 0.585, "hgb_A": 0.315, "pathcrf": 0.1},
    "w_live": {"tcn_B": 0.38, "hgb_B": 0.57, "pathcrf_macro": 0.05},
    "viterbi": {"c_team": 5.0, "c_dead": 30.0, "temp": 1.0},
}

FPS = 25.0
STRIDE = 2  # 12.5 fps for TCN
KMAX = 24   # maximum player slots per team
C_IN = 12   # per-object input channels

PHYS_KEYS = [
    "oob_state", "since_out", "ball_still_run", "dist_line", "in_goal_area", "at_corner",
    "at_centre_spot", "still_at_line", "pl_frac_still", "slow_run", "pl_speed_med", "pl_frac_walk"
]
BASE_TREE_COLS = [
    "vis", "d_home", "d_away", "n_home_3m", "n_away_3m", "n_home_10m", "n_away_10m",
    "ball_speed", "ball_acc", "ball_turn", "ball_absx", "ball_absy", "out", "pl_speed_med",
    "pl_speed_mean", "pl_frac_walk", "pl_frac_run", "home_speed", "away_speed",
    "n_home_vis", "n_away_vis", "home_spread", "away_spread", "since_vis", "until_vis",
    "gap_len"
]
ROLL_COLS = [
    "vis", "pl_speed_med", "pl_frac_walk", "pl_frac_run", "ball_speed", "near_h", "near_a",
    "closer_h", "home_speed", "away_speed"
]
WINDOWS_S = [0.5, 1, 2, 4, 8]


# -----------------------------------------------------------------------------
# 1. True Per-Object 2D-Conv TCN Architecture
# -----------------------------------------------------------------------------

def _vel(P):
    v = np.zeros_like(P)
    v[1:] = (P[1:] - P[:-1]) * FPS / STRIDE
    v = np.nan_to_num(v)
    return np.clip(v, -12.0, 12.0)


def tag_tensor(m, a, b):
    """(n_extra, T') extra channels for the TCN team-level head."""
    sg = m["sig"]
    scale = {
        "since_out": 120.0, "ball_still_run": 10.0, "dist_line": 34.0, "slow_run": 30.0,
        "pl_speed_med": 5.0
    }
    x = np.stack([np.nan_to_num(sg[k][a:b:STRIDE], nan=0.0) / scale.get(k, 1.0) for k in PHYS_KEYS])
    return np.clip(x, -5.0, 5.0).astype(np.float32)


def tensorize_tcn(m, a, b):
    """(C_IN, 1 + 2*KMAX, T') float32 for frames a..b of one half at 12.5 fps."""
    ball = m["ball"][a:b:STRIDE]
    hm = m["home"][a:b:STRIDE][:, :KMAX]
    aw = m["away"][a:b:STRIDE][:, :KMAX]
    T = len(ball)

    def pad(P):
        out = np.full((T, KMAX, 2), np.nan, np.float32)
        out[:, :P.shape[1]] = P
        return out

    objs = np.concatenate([ball[:, None], pad(hm), pad(aw)], 1)  # (T, O, 2)
    O = objs.shape[1]
    vis = ~np.isnan(objs[..., 0])
    bvis = vis[:, :1]
    v = _vel(objs)
    v[~vis] = 0
    v[1:][~vis[:-1]] = 0
    rel = objs - objs[:, :1]
    dist = np.sqrt((rel ** 2).sum(-1))
    relok = vis & bvis

    x = np.zeros((T, O, C_IN), np.float32)
    x[..., 0] = np.nan_to_num(objs[..., 0]) / 52.5
    x[..., 1] = np.nan_to_num(objs[..., 1]) / 34.0
    x[..., 2:4] = v / 10.0
    x[..., 4] = vis
    x[..., 5] = np.where(relok, np.clip(rel[..., 0], -30, 30), 0) / 20.0
    x[..., 6] = np.where(relok, np.clip(rel[..., 1], -30, 30), 0) / 20.0
    x[..., 7] = np.where(relok, np.clip(dist, 0, 40), 40) / 20.0
    x[..., 8] = np.repeat(bvis, O, 1)
    x[:, 0, 9] = 1
    x[:, 1:KMAX + 1, 10] = 1
    x[:, KMAX + 1:, 11] = 1
    x[~vis, :8] = 0
    return np.ascontiguousarray(x.transpose(2, 1, 0)), vis.T.astype(np.float32)


class TCNBlock(nn.Module):
    def __init__(self, ch, dil, drop):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, (1, 3), padding=(0, dil), dilation=(1, dil))
        self.mix = nn.Conv2d(ch, ch, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.mix(self.drop(F.gelu(self.conv(x))))


class PerObjectTCN(nn.Module):
    def __init__(self, hid=48, dils=(1, 2, 4, 8, 16, 32), drop=0.1, n_tag=len(PHYS_KEYS)):
        super().__init__()
        self.n_tag = n_tag
        self.inp = nn.Conv2d(C_IN, hid, 1)
        self.bn = nn.BatchNorm2d(hid)
        self.blocks = nn.Sequential(*[TCNBlock(hid, d, drop) for d in dils])
        self.head = nn.Sequential(
            nn.Conv1d(5 * hid + n_tag, 2 * hid, 1), nn.GELU(),
            nn.Conv1d(2 * hid, 2 * hid, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(2 * hid, 2 * hid, 3, padding=16, dilation=16), nn.GELU(),
            nn.Conv1d(2 * hid, 2, 1)
        )

    def forward(self, x, vis, tag=None):
        e = self.blocks(self.bn(self.inp(x)))
        mk = vis[:, None]
        ball = e[:, :, 0]
        h = e[:, :, 1:KMAX + 1]
        a = e[:, :, KMAX + 1:]
        mh, ma = mk[:, :, 1:KMAX + 1], mk[:, :, KMAX + 1:]
        hsum = (h * mh).sum(2)
        asum = (a * ma).sum(2)
        hmax = torch.where(mh > 0, h, torch.full_like(h, -1e4)).amax(2).clamp(min=-10)
        amax = torch.where(ma > 0, a, torch.full_like(a, -1e4)).amax(2).clamp(min=-10)
        z = torch.cat([ball, hsum / 5, asum / 5, hmax, amax] + ([tag] if tag is not None else []), 1)
        return self.head(z)  # (B, 2, T') logits: alive, home


@torch.no_grad()
def predict_tcn_probabilities(net, m):
    """Predict (T,) p_live and p_home from TCN over all halves."""
    net = net.eval()
    T = len(m["ball"])
    pl = np.full(T, np.nan, np.float32)
    ph = np.full(T, np.nan, np.float32)
    for a, b in m["halves"]:
        x, vis = tensorize_tcn(m, a, b)
        tg = tag_tensor(m, a, b)
        tb = torch.from_numpy(tg[None])
        out = net(torch.from_numpy(x[None]), torch.from_numpy(vis[None]), tb)
        p = torch.sigmoid(out[0]).cpu().numpy()  # (2, T')
        idx = np.arange(a, b)
        src = np.minimum((idx - a) // STRIDE, p.shape[1] - 1)
        pl[a:b] = p[0, src]
        ph[a:b] = p[1, src]
    return pl, ph


# -----------------------------------------------------------------------------
# 2. Asymmetric Past/Future HGB Feature Extraction
# -----------------------------------------------------------------------------

def _nanroll(x, w, shift):
    """Mean of x over a window of w frames, centred at t+shift, ignoring NaN."""
    v = ~np.isnan(x)
    num = uniform_filter1d(np.where(v, x, 0.0), w, mode="nearest", origin=0)
    den = uniform_filter1d(v.astype(float), w, mode="nearest", origin=0)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return np.roll(out, -shift)


def _ffill(x):
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[:np.argmax(~np.isnan(x))] = np.nan if np.isnan(x).all() else out[:np.argmax(~np.isnan(x))]
    return out


def _bfill(x):
    return _ffill(x[::-1])[::-1]


def extract_tree_features(s, a, b):
    """DataFrame of 162 features for frames [a, b) of one half."""
    f = {k: s[k][a:b].astype(np.float32) for k in BASE_TREE_COLS + PHYS_KEYS}
    for k in ("pl_frac_still", "still_at_line", "oob_state"):
        for w_s in (2, 8):
            w = int(w_s * FPS) | 1
            f[f"{k}_p{w_s}"] = _nanroll(f[k], w, -(w // 2))
            f[f"{k}_f{w_s}"] = _nanroll(f[k], w, w // 2)

    dh, da = f["d_home"], f["d_away"]
    diff = dh - da
    f["d_diff"] = diff
    f["d_min"] = np.fmin(dh, da)
    f["near_h"] = np.where(np.isnan(dh), np.nan, (dh < 2.0).astype(float))
    f["near_a"] = np.where(np.isnan(da), np.nan, (da < 2.0).astype(float))
    f["closer_h"] = np.where(np.isnan(diff), np.nan, (diff < 0).astype(float))

    f["last_diff"] = _ffill(diff)
    f["next_diff"] = _bfill(diff)
    f["last_out"] = _ffill(f["out"])
    f["next_out"] = _bfill(f["out"])
    f["last_absx"] = _ffill(f["ball_absx"])
    f["next_absx"] = _bfill(f["ball_absx"])
    f["last_absy"] = _ffill(f["ball_absy"])
    f["next_absy"] = _bfill(f["ball_absy"])

    for k in ROLL_COLS:
        x = f[k]
        for w_s in WINDOWS_S:
            w = int(w_s * FPS) | 1
            f[f"{k}_p{w_s}"] = _nanroll(x, w, -(w // 2))
            f[f"{k}_f{w_s}"] = _nanroll(x, w, w // 2)

    f["t_half"] = np.arange(b - a, dtype=np.float32) / FPS
    return pd.DataFrame(f)


def predict_tree_probabilities(tree_models, m, s):
    """Predict (T,) p_live and p_home from tree models (live_clf, home_clf)."""
    live_clf, home_clf = tree_models
    T = len(m["ball"])
    pl = np.full(T, np.nan, np.float32)
    ph = np.full(T, np.nan, np.float32)

    dfs, idxs = [], []
    for a, b in m["halves"]:
        df_half = extract_tree_features(s, a, b)
        dfs.append(df_half)
        idxs.append(np.arange(a, b))

    X_all = pd.concat(dfs, ignore_index=True)
    idx_all = np.concatenate(idxs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl[idx_all] = live_clf.predict_proba(X_all)[:, 1]
        ph[idx_all] = home_clf.predict_proba(X_all)[:, 1]

    return pl, ph


# -----------------------------------------------------------------------------
# 3. Log-Odds Blending and Calibrated 3-State Viterbi Decoder
# -----------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def blend_probabilities(tcn_p, tree_p, weight_tcn):
    """Blend two probabilities in log-odds space."""
    return 1.0 / (1.0 + np.exp(-(weight_tcn * _logit(tcn_p) + (1.0 - weight_tcn) * _logit(tree_p))))


def blend_three_probabilities(p1, p2, p3, w1, w2, w3):
    """Blend three probability arrays in log-odds space with scalar or per-frame weights."""
    z = w1 * _logit(p1) + w2 * _logit(p2) + w3 * _logit(p3)
    return 1.0 / (1.0 + np.exp(-z))


def load_ensemble_config(path=ENSEMBLE_CFG_PATH):
    """Load external-calibrated multi-model blend and Viterbi weights."""
    cfg = json.loads(json.dumps(_DEFAULT_ENSEMBLE_CFG))
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        for k in ("mode", "pathcrf_readout", "pathcrf_switch_delay_s", "pathcrf_stride_seconds"):
            if k in loaded:
                cfg[k] = loaded[k]
        for k in ("w_home_visible", "w_home_occluded", "w_live", "viterbi"):
            if isinstance(loaded.get(k), dict):
                cfg[k].update(loaded[k])
    return cfg


def viterbi_3state_fast(pl, ph, c_team=4.0, c_dead=25.0, temp=1.0):
    """Exact 3-state Viterbi decoder over log-emissions (H=0, A=1, D=2)."""
    eps = 1e-5
    pl = np.clip(np.nan_to_num(pl, nan=0.5), eps, 1.0 - eps).astype(np.float64)
    ph = np.clip(np.nan_to_num(ph, nan=0.5), eps, 1.0 - eps).astype(np.float64)

    E0 = ((np.log(pl) + np.log(ph)) / temp).tolist()
    E1 = ((np.log(pl) + np.log(1.0 - ph)) / temp).tolist()
    E2 = (np.log(1.0 - pl) / temp).tolist()
    T = len(E0)

    b0 = bytearray(T)
    b1 = bytearray(T)
    b2 = bytearray(T)
    s0, s1, s2 = E0[0], E1[0], E2[0]
    ct, cd = float(c_team), float(c_dead)

    for t in range(1, T):
        # to H (state 0)
        x0, x1, x2 = s0, s1 - ct, s2 - cd
        if x0 >= x1 and x0 >= x2:
            n0, b0[t] = x0, 0
        elif x1 >= x2:
            n0, b0[t] = x1, 1
        else:
            n0, b0[t] = x2, 2

        # to A (state 1)
        x0, x1, x2 = s0 - ct, s1, s2 - cd
        if x1 >= x0 and x1 >= x2:
            n1, b1[t] = x1, 1
        elif x0 >= x2:
            n1, b1[t] = x0, 0
        else:
            n1, b1[t] = x2, 2

        # to D (state 2)
        x0, x1, x2 = s0 - cd, s1 - cd, s2
        if x2 >= x0 and x2 >= x1:
            n2, b2[t] = x2, 2
        elif x0 >= x1:
            n2, b2[t] = x0, 0
        else:
            n2, b2[t] = x1, 1

        s0, s1, s2 = n0 + E0[t], n1 + E1[t], n2 + E2[t]

    path = np.zeros(T, np.int8)
    st = int(np.argmax([s0, s1, s2]))
    backs = (b0, b1, b2)
    for t in range(T - 1, -1, -1):
        path[t] = st
        if t:
            st = backs[st][t]
    return path


# -----------------------------------------------------------------------------
# 4. Super-Hybrid Match Prediction Entry Point
# -----------------------------------------------------------------------------

def load_super_hybrid_models(models_dir=MODELS_DIR, include_pathcrf=True):
    """Load frozen Variant A/B (TCN + HGB) models and fine-tuned Nova PathCRF weights."""
    tcn_A = PerObjectTCN()
    tcn_A.load_state_dict(torch.load(os.path.join(models_dir, "tcn_variant_a.pt"), map_location="cpu", weights_only=True))
    tcn_A.eval()

    tcn_B = PerObjectTCN()
    tcn_B.load_state_dict(torch.load(os.path.join(models_dir, "tcn_variant_b.pt"), map_location="cpu", weights_only=True))
    tcn_B.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.path.join(models_dir, "hgb_variant_a.pkl"), "rb") as f:
            trees_A = pickle.load(f)
        with open(os.path.join(models_dir, "hgb_variant_b.pkl"), "rb") as f:
            trees_B = pickle.load(f)

    nova_pc = None
    ckpt_pc = os.path.join(models_dir, "nova_pathcrf_best_team.pt")
    args_pc = os.path.join(models_dir, "nova_pathcrf_args.json")
    if include_pathcrf and os.path.isfile(ckpt_pc):
        from src.models.nova_pathcrf import load_nova_pathcrf_model
        nova_pc = load_nova_pathcrf_model(ckpt_pc, args_pc)

    return {
        "tcn_A": tcn_A,
        "tcn_B": tcn_B,
        "trees_A": trees_A,
        "trees_B": trees_B,
        "nova_pathcrf": nova_pc,
    }


def predict_super_hybrid(
    t_data,
    fused_ball_x,
    fused_ball_y,
    blocks,
    models=None,
    events=None,
    apply_domain_priors=True,
    c_team=None,
    c_dead=None,
    w_live=None,
    w_poss=None,
    pathcrf_probs=None,
    ensemble_cfg=None,
):
    """Execute complete 3-model Super-Hybrid inference on a match.

    Args:
        t_data: tactical tracking dictionary (with player_x, player_y, team_of_slot, pitch).
        fused_ball_x: fused ball x (including 2,943 recovered broadcast frames), normalized 0..1.
        fused_ball_y: fused ball y, normalized 0..1.
        blocks: list of (half_id, start_frame, last_frame).
        models: dictionary of loaded models (or loaded automatically).
        events: Smart Tagging events for domain priors.
        apply_domain_priors: if True, applies pre-kickoff gate (Smart Tagging Kick Off (Start) window + ball-to-center distance, with <3.5m ball fallback when events=None).
        pathcrf_probs: optional precomputed Nova PathCRF curve dict from build_nova_pathcrf_signals.
        ensemble_cfg: optional ensemble weight dictionary (defaults to config/ensemble_weights.json).
    """
    cfg = ensemble_cfg if ensemble_cfg is not None else load_ensemble_config()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    if models is None:
        models = load_super_hybrid_models(include_pathcrf=(pathcrf_probs is None))

    L, W = float(t_data["pitch"][0]), float(t_data["pitch"][1])
    n_frames = len(fused_ball_x)

    # Convert to centered coordinates for TCN & signals
    bx_c = (fused_ball_x - 0.5) * L
    by_c = (fused_ball_y - 0.5) * W
    ball_c = np.column_stack([bx_c, by_c]).astype(np.float32)

    team_slots = np.asarray(t_data["team_of_slot"])
    home_slots = np.flatnonzero(team_slots == 0)
    away_slots = np.flatnonzero(team_slots == 1)

    px_raw = np.asarray(t_data["player_x"])
    py_raw = np.asarray(t_data["player_y"])

    home_c = np.zeros((n_frames, len(home_slots), 2), dtype=np.float32)
    home_c[..., 0] = (px_raw[:, home_slots] - 0.5) * L
    home_c[..., 1] = (py_raw[:, home_slots] - 0.5) * W

    away_c = np.zeros((n_frames, len(away_slots), 2), dtype=np.float32)
    away_c[..., 0] = (px_raw[:, away_slots] - 0.5) * L
    away_c[..., 1] = (py_raw[:, away_slots] - 0.5) * W

    halves = [(start, last + 1) for _, start, last in blocks]
    m_dict = {
        "ball": ball_c,
        "home": home_c,
        "away": away_c,
        "halves": halves,
    }

    # 1. Compute per-frame signals with slot-switch speed filter (MAX_PLAYER_SPEED = 12.0)
    s = team_signals.compute(m_dict)
    m_dict["sig"] = s

    # 2. Extract tree features ONCE for both Variant A and B
    dfs, idxs = [], []
    for a, b in m_dict["halves"]:
        df_half = extract_tree_features(s, a, b)
        dfs.append(df_half)
        idxs.append(np.arange(a, b))
    X_all = pd.concat(dfs, ignore_index=True)
    idx_all = np.concatenate(idxs)

    # 3. Nova PathCRF (ball-free 5 fps SetTransformer + SAB + DynamicSparseCRF)
    from src.models.nova_pathcrf import predict_nova_pathcrf, select_pathcrf_p_home

    if pathcrf_probs is None:
        pc_model = models.get("nova_pathcrf")
        pathcrf_probs = predict_nova_pathcrf(
            t_data,
            blocks,
            model=pc_model,
            stride_seconds=float(cfg.get("pathcrf_stride_seconds", 10.0)),
            switch_delay_s=0.0,
            readout="crf_src",
        )
    p_pc_home = select_pathcrf_p_home(
        pathcrf_probs,
        blocks,
        readout=str(cfg.get("pathcrf_readout", "crf_src")),
        switch_delay_s=float(cfg.get("pathcrf_switch_delay_s", 1.0)),
    )
    p_pc_live = np.asarray(pathcrf_probs["p_live_macro"], dtype=np.float32)

    # Restore single-thread determinism for Variant A/B TCN inference
    torch.set_num_threads(1)

    # 4. Variant B + PathCRF Macro: predict P(live)
    B_tl, _ = predict_tcn_probabilities(models["tcn_B"], m_dict)
    live_clf_B = models["trees_B"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B_gl = np.full(n_frames, np.nan, np.float32)
        B_gl[idx_all] = live_clf_B.predict_proba(X_all)[:, 1]

    if w_live is not None:
        p_live = blend_probabilities(B_tl, B_gl, weight_tcn=w_live)
    else:
        wl = cfg["w_live"]
        p_live = blend_three_probabilities(B_tl, B_gl, p_pc_live, wl["tcn_B"], wl["hgb_B"], wl["pathcrf_macro"])

    # 5. Variant A + Nova PathCRF: predict P(home | live) (visibility-gated or uniform)
    _, A_th = predict_tcn_probabilities(models["tcn_A"], m_dict)
    home_clf_A = models["trees_A"][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_gh = np.full(n_frames, np.nan, np.float32)
        A_gh[idx_all] = home_clf_A.predict_proba(X_all)[:, 1]

    if w_poss is not None:
        p_home = blend_probabilities(A_th, A_gh, weight_tcn=w_poss)
    else:
        ball_vis = np.isfinite(fused_ball_x)
        wv = cfg["w_home_visible"]
        wo = cfg["w_home_occluded"]
        wt = np.where(ball_vis, wv["tcn_A"], wo["tcn_A"])
        wh = np.where(ball_vis, wv["hgb_A"], wo["hgb_A"])
        wp = np.where(ball_vis, wv["pathcrf"], wo["pathcrf"])
        p_home = blend_three_probabilities(A_th, A_gh, p_pc_home, wt, wh, wp)

    # 6. Pre-kickoff gate on P(live): Law 8 dead ball prior to opening touch
    if apply_domain_priors:
        from src.features import detect_kickoff_frame
        pre_kickoff = detect_kickoff_frame(fused_ball_x, fused_ball_y, blocks, (L, W), events=events)
        p_live[pre_kickoff] = 0.0

    # 7. Decode whole-match with calibrated 3-state Viterbi
    vit = cfg["viterbi"]
    ct = float(vit["c_team"] if c_team is None else c_team)
    cd = float(vit["c_dead"] if c_dead is None else c_dead)
    tp = float(vit.get("temp", 1.0))

    labels = np.full(n_frames, 3, dtype=np.int8)  # 3 = X (outside halves)
    for _, start, last in blocks:
        stop = last + 1
        path = viterbi_3state_fast(p_live[start:stop], p_home[start:stop], c_team=ct, c_dead=cd, temp=tp)
        labels[start:stop] = path

    return {
        "labels": labels,
        "p_live": p_live,
        "p_home": p_home,
        "p_pathcrf_home": p_pc_home,
        "p_pathcrf_live": p_pc_live,
        "signals": s,
    }
