"""External weight sweep for integrating Nova PathCRF into Super-Hybrid.

Evaluates on the 3 unseen Metrica Sample Games (`metrica_1`, `metrica_2`, `metrica_3`,
harmonized to CHALLENGE.md's controlled-possession definition and degraded with
realistic tracking noise v3 matching BAR-ATH's ~41% missing ball and 20s slot swaps).
These 3 matches are out-of-sample with respect to model training (`Variant A/B` and
`Nova PathCRF`), while evaluating the winning grid configuration on all 3 matches
introduces ensemble-weight selection optimism (`+0.0287` spellF1). Therefore this script
also computes a strict nested 3-fold Leave-One-Match-Out (LOMO) evaluation (`+0.0295`
mean spellF1 gain). Zero BAR-ATH labels or ground truth are read or referenced anywhere.

Performs three structured sweeps over external Metrica matches + strict nested LOMO:
  Stage A: PathCRF readout (`crf_src`, `crf_mid`, `macro`, `crf_macro_50`) and switch delay
  Stage B: 3-model P(home | live) weights (TCN + HGB + PathCRF) for both uniform
           simplices and ball-visibility-gated (visible vs occluded) ensembles
  Stage C: 3-model P(live) weight + Viterbi transition penalty check around (4.0, 25.0)
  LOMO:    Strict 3-fold Leave-One-Match-Out repeating Stages A/B/C on 2 matches per fold

Outputs:
  - config/ensemble_weights.json  (frozen weights read by src/models/super_hybrid.py)
  - reports/pathcrf_weight_sweep.md (summary and nested 3-fold LOMO table)
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm  # noqa: E402
from src import team_signals  # noqa: E402
from src.external.metrica import load_metrica_game  # noqa: E402
from src.models.nova_pathcrf import (  # noqa: E402
    load_nova_pathcrf_model,
    predict_nova_pathcrf,
    select_pathcrf_p_home,
)
from src.models.super_hybrid import (  # noqa: E402
    extract_tree_features,
    load_super_hybrid_models,
    predict_tcn_probabilities,
    viterbi_3state_fast,
)
from tools.train_super_hybrid import degrade_v3  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "external", "cache")
CFG_OUT = os.path.join(ROOT, "config", "ensemble_weights.json")
REPORT_OUT = os.path.join(ROOT, "reports", "pathcrf_weight_sweep.md")


def _logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def blend3_logodds(p1, p2, p3, w1, w2, w3):
    z = w1 * _logit(p1) + w2 * _logit(p2) + w3 * _logit(p3)
    return 1.0 / (1.0 + np.exp(-z))


def _prepare_degraded_metrica(game_no, sh_models, pc_model, seed=0, stride_seconds=10.0, force=False):
    cache_path = os.path.join(CACHE_DIR, f"sweep_probs_metrica_{game_no}_v3_s{seed}_str{int(stride_seconds)}.npz")
    m = load_metrica_game(game_no, use_cache=True, harmonize=True)
    L, W = float(m.pitch[0]), float(m.pitch[1])

    if not force and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            return {
                "name": m.name,
                "blocks": m.blocks,
                "labels": m.labels,
                "ball_vis": z["ball_vis"],
                "b_tcn_live": z["b_tcn_live"],
                "b_hgb_live": z["b_hgb_live"],
                "a_tcn_home": z["a_tcn_home"],
                "a_hgb_home": z["a_hgb_home"],
                "pc_live_macro": z["pc_live_macro"],
                "p_home_crf_src": z["p_home_crf_src"],
                "p_home_crf_mid": z["p_home_crf_mid"],
                "p_home_macro": z["p_home_macro"],
            }

    home_s = m.team_of_slot == 0
    away_s = m.team_of_slot == 1
    m_cent = {
        "name": m.name,
        "ball": np.column_stack([(m.ball_x - 0.5) * L, (m.ball_y - 0.5) * W]).astype(np.float32),
        "home": np.stack([(m.player_x[:, home_s] - 0.5) * L, (m.player_y[:, home_s] - 0.5) * W], -1).astype(np.float32),
        "away": np.stack([(m.player_x[:, away_s] - 0.5) * L, (m.player_y[:, away_s] - 0.5) * W], -1).astype(np.float32),
        "label": np.where(m.labels == pm.H, 0, np.where(m.labels == pm.A, 1, 2)).astype(np.int8),
        "halves": [(s, e + 1) for _, s, e in m.blocks],
    }
    deg = degrade_v3(m_cent, seed=seed)
    all_p = np.concatenate([deg["home"], deg["away"]], axis=1)
    t_deg = {
        "pitch": m.pitch,
        "team_of_slot": np.array([0] * deg["home"].shape[1] + [1] * deg["away"].shape[1]),
        "player_x": (all_p[..., 0] / L + 0.5).astype(np.float32),
        "player_y": (all_p[..., 1] / W + 0.5).astype(np.float32),
    }

    m_dict = {
        "ball": deg["ball"],
        "home": deg["home"],
        "away": deg["away"],
        "halves": m_cent["halves"],
    }
    sig = team_signals.compute(m_dict)
    m_dict["sig"] = sig

    dfs, idxs = [], []
    for a, b in m_dict["halves"]:
        dfs.append(extract_tree_features(sig, a, b))
        idxs.append(np.arange(a, b))
    X_all = pd.concat(dfs, ignore_index=True)
    idx_all = np.concatenate(idxs)

    n = len(m.labels)
    b_tcn_live, _ = predict_tcn_probabilities(sh_models["tcn_B"], m_dict)
    _, a_tcn_home = predict_tcn_probabilities(sh_models["tcn_A"], m_dict)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_hgb_live = np.full(n, np.nan, np.float32)
        a_hgb_home = np.full(n, np.nan, np.float32)
        b_hgb_live[idx_all] = sh_models["trees_B"][0].predict_proba(X_all)[:, 1]
        a_hgb_home[idx_all] = sh_models["trees_A"][1].predict_proba(X_all)[:, 1]

    print(f"[{m.name} v3 s={seed}] Running Nova PathCRF (stride={stride_seconds}s)...", flush=True)
    pc = predict_nova_pathcrf(
        t_deg,
        m.blocks,
        model=pc_model,
        stride_seconds=stride_seconds,
        switch_delay_s=0.0,
        readout="crf_src",
        verbose=True,
    )
    ball_vis = np.isfinite(deg["ball"][:, 0])

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ball_vis=ball_vis,
        b_tcn_live=b_tcn_live.astype(np.float32),
        b_hgb_live=b_hgb_live.astype(np.float32),
        a_tcn_home=a_tcn_home.astype(np.float32),
        a_hgb_home=a_hgb_home.astype(np.float32),
        pc_live_macro=pc["p_live_macro"].astype(np.float32),
        p_home_crf_src=pc["p_home_crf_src"].astype(np.float32),
        p_home_crf_mid=pc["p_home_crf_mid"].astype(np.float32),
        p_home_macro=pc["p_home_macro"].astype(np.float32),
    )
    return {
        "name": m.name,
        "blocks": m.blocks,
        "labels": m.labels,
        "ball_vis": ball_vis,
        "b_tcn_live": b_tcn_live.astype(np.float32),
        "b_hgb_live": b_hgb_live.astype(np.float32),
        "a_tcn_home": a_tcn_home.astype(np.float32),
        "a_hgb_home": a_hgb_home.astype(np.float32),
        "pc_live_macro": pc["p_live_macro"].astype(np.float32),
        "p_home_crf_src": pc["p_home_crf_src"].astype(np.float32),
        "p_home_crf_mid": pc["p_home_crf_mid"].astype(np.float32),
        "p_home_macro": pc["p_home_macro"].astype(np.float32),
    }


def _eval_config(items, readout, delay_s, w_vis, w_occ, w_live, c_team=4.0, c_dead=25.0):
    per_match = []
    for it in items:
        blocks = it["blocks"]
        p_live = blend3_logodds(
            it["b_tcn_live"], it["b_hgb_live"], it["pc_live_macro"], w_live[0], w_live[1], w_live[2]
        )
        p_pc = select_pathcrf_p_home(it, blocks, readout=readout, switch_delay_s=delay_s)
        vis_b = it["ball_vis"]
        wt = np.where(vis_b, w_vis[0], w_occ[0])
        wh = np.where(vis_b, w_vis[1], w_occ[1])
        wp = np.where(vis_b, w_vis[2], w_occ[2])
        p_home = blend3_logodds(it["a_tcn_home"], it["a_hgb_home"], p_pc, wt, wh, wp)

        pred = np.full(len(it["labels"]), pm.X, dtype=np.int8)
        for _, start, last in blocks:
            pred[start : last + 1] = viterbi_3state_fast(
                p_live[start : last + 1], p_home[start : last + 1], c_team=c_team, c_dead=c_dead
            )
        ev = pm.evaluate(it["labels"], pred, bootstrap_n=0)
        per_match.append({
            "name": it["name"],
            "spell_f1": float(ev["spell_f1_mean"]),
            "macro_f1": float(ev["macro_f1"]),
            "f1_D": float(ev["per_class"]["D"]["f1"]),
            "team_acc": float(ev["team_acc_both_alive"]),
        })
    return {
        "mean_spell": float(np.mean([r["spell_f1"] for r in per_match])),
        "mean_macro": float(np.mean([r["macro_f1"] for r in per_match])),
        "mean_f1_D": float(np.mean([r["f1_D"] for r in per_match])),
        "mean_team_acc": float(np.mean([r["team_acc"] for r in per_match])),
        "per_match": per_match,
    }


def _select_grid(eval_items, base_live):
    """Run Stage A -> B1/B2 -> C grid search on `eval_items`."""
    # Stage A: Readout & switch delay sweep (evaluated at 20% uniform PathCRF blend + standalone)
    stage_a_rows = []
    for ro in ("crf_src", "crf_mid", "macro", "crf_macro_50"):
        for d_s in (0.0, 0.5, 1.0):
            r20 = _eval_config(eval_items, ro, d_s, (0.60, 0.20, 0.20), (0.60, 0.20, 0.20), base_live)
            r100 = _eval_config(eval_items, ro, d_s, (0.00, 0.00, 1.00), (0.00, 0.00, 1.00), base_live)
            stage_a_rows.append((ro, d_s, r20["mean_spell"], r20["mean_macro"], r100["mean_spell"], r100["mean_team_acc"]))

    stage_a_rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
    best_ro, best_delay = stage_a_rows[0][0], stage_a_rows[0][1]

    # Stage B1: Uniform 3-model P(home | live) simplex sweep
    unif_rows = []
    for wp in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00):
        ratios = (0.75,) if wp == 1.00 else (0.50, 0.65, 0.75, 0.85)
        for rt in ratios:
            wt = round((1.0 - wp) * rt, 6)
            wh = round((1.0 - wp) * (1.0 - rt), 6)
            res = _eval_config(eval_items, best_ro, best_delay, (wt, wh, wp), (wt, wh, wp), base_live)
            unif_rows.append(((wt, wh, wp), res))
    unif_rows.sort(key=lambda x: (x[1]["mean_spell"], x[1]["mean_macro"]), reverse=True)

    # Stage B2: Ball-visibility-gated 3-model P(home | live) sweep (vis vs occluded)
    gated_rows = []
    for wp_v in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25):
        for wp_o in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
            for rt in (0.65, 0.75, 0.85):
                wv = (round((1.0 - wp_v) * rt, 6), round((1.0 - wp_v) * (1.0 - rt), 6), wp_v)
                wo = (round((1.0 - wp_o) * rt, 6), round((1.0 - wp_o) * (1.0 - rt), 6), wp_o)
                res = _eval_config(eval_items, best_ro, best_delay, wv, wo, base_live)
                gated_rows.append((wv, wo, res))
    gated_rows.sort(key=lambda x: (x[2]["mean_spell"], x[2]["mean_macro"]), reverse=True)

    best_unif_w, best_unif_res = unif_rows[0]
    best_gate_wv, best_gate_wo, best_gate_res = gated_rows[0]

    if (best_gate_res["mean_spell"], best_gate_res["mean_macro"]) > (best_unif_res["mean_spell"], best_unif_res["mean_macro"]):
        win_wv, win_wo, win_b_res = best_gate_wv, best_gate_wo, best_gate_res
        win_mode = "visibility_gated"
    else:
        win_wv, win_wo, win_b_res = best_unif_w, best_unif_w, best_unif_res
        win_mode = "uniform"

    # Stage C: P(live) weight & Viterbi cost check around (4.0, 25.0)
    stage_c_rows = []
    for w_lp in (0.00, 0.05, 0.10, 0.15):
        for rl in (0.40, 0.50, 0.60):
            w_l = (round((1.0 - w_lp) * rl, 6), round((1.0 - w_lp) * (1.0 - rl), 6), w_lp)
            for ct in (3.0, 4.0, 5.0):
                for cd in (20.0, 25.0, 30.0):
                    res = _eval_config(eval_items, best_ro, best_delay, win_wv, win_wo, w_l, c_team=ct, c_dead=cd)
                    stage_c_rows.append((w_l, ct, cd, res))
    stage_c_rows.sort(key=lambda x: (x[3]["mean_spell"], x[3]["mean_macro"]), reverse=True)
    win_wl, win_ct, win_cd, final_res = stage_c_rows[0]

    return {
        "best_ro": best_ro,
        "best_delay": best_delay,
        "win_mode": win_mode,
        "win_wv": win_wv,
        "win_wo": win_wo,
        "win_wl": win_wl,
        "win_ct": win_ct,
        "win_cd": win_cd,
        "final_res": final_res,
        "stage_a_rows": stage_a_rows,
        "unif_rows": unif_rows,
        "gated_rows": gated_rows,
        "stage_c_rows": stage_c_rows,
        "best_unif_res": best_unif_res,
        "win_b_res": win_b_res,
    }


def _run_nested_lomo(items, base_live, base_res):
    """Strict 3-fold Leave-One-Match-Out across Metrica 1-3 (select on 2, test on 3rd)."""
    fold_rows = []
    for held_idx in range(len(items)):
        tr_items = [items[j] for j in range(len(items)) if j != held_idx]
        te_items = [items[held_idx]]
        sel = _select_grid(tr_items, base_live)
        te_res = _eval_config(
            te_items,
            sel["best_ro"],
            sel["best_delay"],
            sel["win_wv"],
            sel["win_wo"],
            sel["win_wl"],
            c_team=sel["win_ct"],
            c_dead=sel["win_cd"],
        )
        base_m = base_res["per_match"][held_idx]
        fold_rows.append({
            "name": te_items[0]["name"],
            "baseline_spell": base_m["spell_f1"],
            "baseline_macro": base_m["macro_f1"],
            "lomo_spell": te_res["mean_spell"],
            "lomo_macro": te_res["mean_macro"],
            "lomo_f1_D": te_res["mean_f1_D"],
            "gain_spell": te_res["mean_spell"] - base_m["spell_f1"],
            "sel_ro": sel["best_ro"],
            "sel_delay": sel["best_delay"],
            "sel_wv": sel["win_wv"],
            "sel_wo": sel["win_wo"],
            "sel_wl": sel["win_wl"],
            "sel_ct": sel["win_ct"],
            "sel_cd": sel["win_cd"],
        })
    return {
        "fold_rows": fold_rows,
        "mean_spell": float(np.mean([r["lomo_spell"] for r in fold_rows])),
        "mean_macro": float(np.mean([r["lomo_macro"] for r in fold_rows])),
        "mean_f1_D": float(np.mean([r["lomo_f1_D"] for r in fold_rows])),
        "mean_gain": float(np.mean([r["gain_spell"] for r in fold_rows])),
    }


def main():
    sh_models = load_super_hybrid_models()
    pc_model = load_nova_pathcrf_model()

    items = [_prepare_degraded_metrica(g, sh_models, pc_model, seed=0, stride_seconds=10.0) for g in (1, 2, 3)]
    base_live = (0.50, 0.50, 0.00)

    # Baseline references
    res_base_2m = _eval_config(items, "crf_src", 0.0, (0.75, 0.25, 0.00), (0.75, 0.25, 0.00), base_live)
    print(f"Baseline (2-model TCN=0.75, HGB=0.25, PathCRF=0.00): mean spellF1={res_base_2m['mean_spell']:.4f}, macroF1={res_base_2m['mean_macro']:.4f}")

    # Full 3-match grid selection (weights frozen for inference)
    full = _select_grid(items, base_live)
    best_ro, best_delay = full["best_ro"], full["best_delay"]
    win_mode, win_wv, win_wo = full["win_mode"], full["win_wv"], full["win_wo"]
    win_wl, win_ct, win_cd = full["win_wl"], full["win_ct"], full["win_cd"]
    final_res = full["final_res"]
    best_unif_res = full["best_unif_res"]
    stage_a_rows = full["stage_a_rows"]
    unif_rows = full["unif_rows"]
    gated_rows = full["gated_rows"]
    stage_c_rows = full["stage_c_rows"]

    for ro, d_s, sp20, _, _, tacc in stage_a_rows:
        print(f"  [Stage A] {ro:12s} delay={d_s:.1f}s -> blend20 spell={sp20:.4f} | pure_pc team_acc={tacc:.4f}")
    print(f"Selected Stage A readout={best_ro}, switch_delay_s={best_delay}")

    # Strict Nested 3-Fold LOMO (unbiased weight-selection gain)
    lomo = _run_nested_lomo(items, base_live, res_base_2m)
    grid_optimism = final_res["mean_spell"] - lomo["mean_spell"]
    print(
        f"NESTED 3-FOLD LOMO: mean spellF1={lomo['mean_spell']:.4f} "
        f"(vs baseline {res_base_2m['mean_spell']:.4f}, +{lomo['mean_gain']:.4f}), "
        f"macroF1={lomo['mean_macro']:.4f}, f1_D={lomo['mean_f1_D']:.4f} | "
        f"in-grid optimism=+{grid_optimism:.4f}"
    )

    # Save frozen config (inference keys identical to pre-audit)
    cfg = {
        "benchmark": "3 Metrica sample matches degraded with noise v3 (seed=0; model-training out-of-sample, weights grid-selected on these 3 matches)",
        "mode": win_mode,
        "pathcrf_readout": best_ro,
        "pathcrf_switch_delay_s": best_delay,
        "pathcrf_stride_seconds": 10.0,
        "w_home_visible": {"tcn_A": win_wv[0], "hgb_A": win_wv[1], "pathcrf": win_wv[2]},
        "w_home_occluded": {"tcn_A": win_wo[0], "hgb_A": win_wo[1], "pathcrf": win_wo[2]},
        "w_live": {"tcn_B": win_wl[0], "hgb_B": win_wl[1], "pathcrf_macro": win_wl[2]},
        "viterbi": {"c_team": win_ct, "c_dead": win_cd, "temp": 1.0},
        "external_benchmark_summary": {
            "baseline_2model_spell_f1": round(res_base_2m["mean_spell"], 4),
            "baseline_2model_macro_f1": round(res_base_2m["mean_macro"], 4),
            "nested_lomo_3model_spell_f1": round(lomo["mean_spell"], 4),
            "nested_lomo_3model_macro_f1": round(lomo["mean_macro"], 4),
            "nested_lomo_delta_spell_f1": round(lomo["mean_gain"], 4),
            "best_uniform_spell_f1": round(best_unif_res["mean_spell"], 4),
            "selected_3model_in_grid_spell_f1": round(final_res["mean_spell"], 4),
            "selected_3model_in_grid_macro_f1": round(final_res["mean_macro"], 4),
            "in_grid_delta_spell_f1": round(final_res["mean_spell"] - res_base_2m["mean_spell"], 4),
            "grid_selection_optimism_spell_f1": round(grid_optimism, 4),
        },
    }
    os.makedirs(os.path.dirname(CFG_OUT), exist_ok=True)
    with open(CFG_OUT, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {CFG_OUT}")
    print(
        f"SELECTED [{win_mode}]: nested_LOMO={lomo['mean_spell']:.4f} (+{lomo['mean_gain']:.4f}), "
        f"in-grid={final_res['mean_spell']:.4f} (vs baseline {res_base_2m['mean_spell']:.4f}, "
        f"+{final_res['mean_spell'] - res_base_2m['mean_spell']:.4f}), macroF1={final_res['mean_macro']:.4f}"
    )

    # Write concise sweep & nested LOMO summary report
    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    lines = [
        "# Ensemble Weight Calibration (`TCN + HGB + Nova PathCRF`)",
        "",
        f"Evaluated on external matches (**Metrica Sample Games 1, 2, 3** with harmonized controlled-possession labels via `src/external/harmonize.py` and synthetic `noise v3` matching ~41% missing ball and 20s slot swaps; `seed=0`). None of these 3 matches were used to train `PerObjectTCN`, `HGB`, or `Youssef16Ahmed/nova-pathcrf-finetuned`. Because `config/ensemble_weights.json` was selected on all 3 matches (`{final_res['mean_spell']:.4f}` in-grid, `+{grid_optimism:.4f}` selection optimism), the **strict nested 3-fold LOMO** score (**`{lomo['mean_spell']:.4f}`**, **`{lomo['mean_gain']:+.4f}`** over the `{res_base_2m['mean_spell']:.4f}` 2-model baseline) measures unbiased out-of-sample weight generalization.",
        "",
        "## 1. Selected Configuration (`config/ensemble_weights.json`)",
        "",
        f"- **Blend Mode:** `{win_mode}` (`readout = {best_ro}`, `switch_delay_s = {best_delay}`, `stride_seconds = 10.0`)",
        f"- **`P(home | live)` (Ball Visible):** `TCN_A = {win_wv[0]:.4f}`, `HGB_A = {win_wv[1]:.4f}`, `Nova_PathCRF = {win_wv[2]:.4f}`",
        f"- **`P(home | live)` (Ball Missing):** `TCN_A = {win_wo[0]:.4f}`, `HGB_A = {win_wo[1]:.4f}`, `Nova_PathCRF = {win_wo[2]:.4f}`",
        f"- **`P(live)`:** `TCN_B = {win_wl[0]:.4f}`, `HGB_B = {win_wl[1]:.4f}`, `Nova_PathCRF_Macro = {win_wl[2]:.4f}`",
        f"- **Viterbi Penalties:** `c_team = {win_ct}`, `c_dead = {win_cd}`",
        "",
        "## 2. Strict Nested 3-Fold Leave-One-Match-Out",
        "",
        "Each fold runs the full weight grid selection on 2 Metrica matches and evaluates blindly on the 3rd held-out match:",
        "",
        "| Held-Out Match | 2-Model Baseline `spellF1` | Nested LOMO `spellF1` | Nested LOMO `macroF1` | `spellF1` Gain | Fold Selection `(readout, delay, c_team/c_dead)` |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for fr in lomo["fold_rows"]:
        lines.append(
            f"| `{fr['name']}` | {fr['baseline_spell']:.4f} | **{fr['lomo_spell']:.4f}** | {fr['lomo_macro']:.4f} | **{fr['gain_spell']:+.4f}** | `{fr['sel_ro']}`, `{fr['sel_delay']:.1f}s`, `({fr['sel_ct']:.0f}/{fr['sel_cd']:.0f})` |"
        )
    lines.append(
        f"| **Mean (3 Held-Out Folds)** | **{res_base_2m['mean_spell']:.4f}** | **{lomo['mean_spell']:.4f}** | **{lomo['mean_macro']:.4f}** | **{lomo['mean_gain']:+.4f}** | Mean `f1_D = {lomo['mean_f1_D']:.4f}` |"
    )

    with open(REPORT_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
