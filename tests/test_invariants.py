"""Invariant and integrity test suite for Decompass possession pipeline."""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from games import GAMES, GAME, half_video_blocks
from src.broadcast_align import build_broadcast_alignment
from src.cache_atd import build_cache
from src.features import detect_kickoff_frame
from src.models.nova_pathcrf import build_nova_pathcrf_signals, predict_nova_pathcrf
from src.models.super_hybrid import load_ensemble_config, load_super_hybrid_models, predict_super_hybrid
from starter.load_smart_tagging import load as load_smart_tagging


def test_submission_file_format_valid():
    csv_path = os.path.join(ROOT, "submission", "Decompass.csv")
    if os.path.isfile(csv_path):
        labels = pm.read_label_csv(csv_path)
        err = pm.validate_submission(labels, GAMES[GAME]["n_video_frames"])
        assert err is None, f"Submission file failed validation: {err}"


@pytest.fixture(scope="module")
def super_hybrid_artifacts():
    blocks = half_video_blocks(GAME)
    t_data = build_cache(game=GAME, source="tactical")
    b_align = build_broadcast_alignment(game=GAME)
    events = load_smart_tagging(game=GAME)
    ens_cfg = load_ensemble_config()
    pc_probs = build_nova_pathcrf_signals(
        t_data, blocks, game=GAME, stride_seconds=float(ens_cfg["pathcrf_stride_seconds"]), force=False
    )
    models = load_super_hybrid_models(include_pathcrf=True)
    res = predict_super_hybrid(
        t_data,
        b_align["fused_ball_x"],
        b_align["fused_ball_y"],
        blocks,
        models=models,
        events=events,
        pathcrf_probs=pc_probs,
        ensemble_cfg=ens_cfg,
    )
    pre_ko = detect_kickoff_frame(
        b_align["fused_ball_x"], b_align["fused_ball_y"], blocks, t_data["pitch"], events=events
    )
    return {
        "blocks": blocks,
        "t_data": t_data,
        "b_align": b_align,
        "events": events,
        "models": models,
        "pred": res["labels"],
        "pre_ko": pre_ko,
    }


def test_super_hybrid_dimensions_and_labels(super_hybrid_artifacts):
    pred = super_hybrid_artifacts["pred"]
    n_expected = GAMES[GAME]["n_video_frames"]
    assert len(pred) == n_expected, f"Expected {n_expected} frames, got {len(pred)}"
    assert np.all(np.isin(pred, [pm.H, pm.A, pm.D, pm.X])), "Invalid labels outside {H, A, D, X}"


def test_super_hybrid_ungraded_intervals_strictly_x(super_hybrid_artifacts):
    pred = super_hybrid_artifacts["pred"]
    blocks = super_hybrid_artifacts["blocks"]
    graded = np.zeros(len(pred), dtype=bool)
    for _, s, l in blocks:
        graded[s : l + 1] = True
    assert np.all(pred[~graded] == pm.X), "Ungraded frames must be labeled X"
    assert not np.any(pred[graded] == pm.X), "Graded frames must never be labeled X"


def test_super_hybrid_pre_kickoff_is_dead(super_hybrid_artifacts):
    pred = super_hybrid_artifacts["pred"]
    pre_ko = super_hybrid_artifacts["pre_ko"]
    assert np.all(pre_ko[110808:113742]), "2nd-half pre-kickoff interval 110808..113741 must be masked"
    assert np.all(pred[pre_ko] == pm.D), "Pre-kickoff frames must be strictly dead ball (D)"


def test_super_hybrid_determinism_and_submission_match(super_hybrid_artifacts):
    t_data = super_hybrid_artifacts["t_data"]
    b_align = super_hybrid_artifacts["b_align"]
    models = super_hybrid_artifacts["models"]
    events = super_hybrid_artifacts["events"]

    sub_end = 20000
    sub_t = {
        "pitch": t_data["pitch"],
        "team_of_slot": t_data["team_of_slot"],
        "player_x": t_data["player_x"][:sub_end],
        "player_y": t_data["player_y"][:sub_end],
    }
    sub_bx = b_align["fused_ball_x"][:sub_end]
    sub_by = b_align["fused_ball_y"][:sub_end]
    sub_blocks = [(1, 11185, sub_end - 1)]

    r1 = predict_super_hybrid(sub_t, sub_bx, sub_by, sub_blocks, models=models, events=events)
    r2 = predict_super_hybrid(sub_t, sub_bx, sub_by, sub_blocks, models=models, events=events)
    assert np.array_equal(r1["labels"], r2["labels"]), "Ensemble labels must be deterministic"
    assert np.allclose(r1["p_live"], r2["p_live"], equal_nan=True), "Ensemble P(live) must be deterministic"

    csv_path = os.path.join(ROOT, "submission", "Decompass.csv")
    if os.path.isfile(csv_path):
        file_codes = pm.label_codes(pm.read_label_csv(csv_path))
        assert np.array_equal(super_hybrid_artifacts["pred"], file_codes), (
            "submission/Decompass.csv does not match current pipeline output"
        )


def test_nova_pathcrf_probabilities_and_determinism(super_hybrid_artifacts):
    t_data = super_hybrid_artifacts["t_data"]
    pc_model = super_hybrid_artifacts["models"]["nova_pathcrf"]
    assert pc_model is not None, "Nova PathCRF model failed to load"

    sub_end = 13685
    sub_t = {
        "pitch": t_data["pitch"],
        "team_of_slot": t_data["team_of_slot"],
        "player_x": t_data["player_x"][:sub_end],
        "player_y": t_data["player_y"][:sub_end],
    }
    sub_blocks = [(1, 11185, sub_end - 1)]
    r1 = predict_nova_pathcrf(sub_t, sub_blocks, model=pc_model, stride_seconds=10.0)
    r2 = predict_nova_pathcrf(sub_t, sub_blocks, model=pc_model, stride_seconds=10.0)

    for key in ("p_home_crf_src", "p_home_crf_mid", "p_home_macro", "p_live_macro"):
        arr1, arr2 = r1[key][11185:sub_end], r2[key][11185:sub_end]
        assert np.all(np.isfinite(arr1)), f"Non-finite values in Nova PathCRF {key}"
        assert np.all((arr1 >= 0.0) & (arr1 <= 1.0)), f"Out-of-range probabilities in {key}"
        assert np.array_equal(arr1, arr2), f"Nova PathCRF {key} is not bit-for-bit deterministic"
