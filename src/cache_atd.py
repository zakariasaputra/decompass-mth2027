"""Fast parser and binary (.npz) cache for Metrica Automatic Tracking Data (ATD).

Reading the 195 MB raw ATD text file line-by-line in pure Python takes ~60 seconds.
By normalizing separators (':' and ';' -> ',') and parsing via pandas' C engine once,
we convert the entire 199,319-frame dataset into a compressed NumPy archive in `data/cache/`
in a few seconds, and subsequent loads take < 0.3 seconds.
"""

import io
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DEFAULT_SOURCE, GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402


def parse_atd_metadata(meta_path):
    """Extract team slot assignments, team names, and first-half attack directions from XML."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")

    team_ids = [t.get("id") for t in teams]
    team_names = [t.findtext("Name") or "" for t in teams]

    attack_dir = []
    for t in teams:
        direction = 0
        for param in t.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                val = (param.findtext("Value") or "").strip().lower()
                if val == "left_to_right":
                    direction = 1
                elif val == "right_to_left":
                    direction = -1
        attack_dir.append(direction)

    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries found")

    team_of_slot = np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)
    return {
        "team_of_slot": team_of_slot,
        "team_names": np.array(team_names),
        "attack_dir_first_half": np.array(attack_dir, dtype=np.int8),
    }


def cache_path(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Path to the cached .npz file inside data/cache/ (gitignored via data/*)."""
    root = data_dir or DATA_DIR
    cache_dir = os.path.join(root, "cache")
    return os.path.join(cache_dir, f"{game}_{source}_atd.npz")


def build_cache(game=GAME, data_dir=None, source=DEFAULT_SOURCE, force=False):
    """Build or load the compressed .npz cache for the given ATD source."""
    out_path = cache_path(game, data_dir, source)
    if os.path.isfile(out_path) and not force:
        t0 = time.time()
        data = dict(np.load(out_path, allow_pickle=False))
        print(f"[cache_atd] Loaded cached {source} ATD from {os.path.relpath(out_path, ROOT)} "
              f"in {time.time() - t0:.2f}s")
        return data

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    meta = parse_atd_metadata(meta_path)
    n_slots = len(meta["team_of_slot"])

    print(f"[cache_atd] Parsing raw {source} ATD ({os.path.basename(raw_path)}, "
          f"{n_frames} frames, {n_slots} slots)...")
    t0 = time.time()

    with open(raw_path, "rb") as f:
        raw_bytes = f.read()

    # Normalize ':' and ';' to ',' so the C CSV parser reads all 1 + 3*n_slots + 3 columns at once
    trans_table = bytes.maketrans(b":;", b",,")
    csv_bytes = raw_bytes.translate(trans_table)
    del raw_bytes

    n_cols = 1 + 3 * n_slots + 3
    dtypes = {0: np.int32, **{col: np.float32 for col in range(1, n_cols)}}

    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=None,
        dtype=dtypes,
        na_values=["NaN"],
        keep_default_na=False,
        engine="c",
    )
    del csv_bytes

    frame_ids = df.iloc[:, 0].values
    if len(frame_ids) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError(f"Frame alignment mismatch in {raw_path}: got {len(frame_ids)} rows")

    mat = df.iloc[:, 1:].values  # shape (n_frames, 3*n_slots + 3), float32
    players = mat[:, : 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = mat[:, 3 * n_slots :]  # shape (n_frames, 3)

    player_x = np.ascontiguousarray(players[:, :, 0])
    player_y = np.ascontiguousarray(players[:, :, 1])
    player_s = np.ascontiguousarray(players[:, :, 2])

    ball_x = np.ascontiguousarray(ball[:, 0])
    ball_y = np.ascontiguousarray(ball[:, 1])
    ball_s = np.ascontiguousarray(ball[:, 2])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pitch = np.array(GAMES[game]["pitch"], dtype=np.float64)

    np.savez(
        out_path,
        player_x=player_x,
        player_y=player_y,
        player_s=player_s,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_s=ball_s,
        team_of_slot=meta["team_of_slot"],
        team_names=meta["team_names"],
        attack_dir_first_half=meta["attack_dir_first_half"],
        pitch=pitch,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[cache_atd] Wrote {os.path.relpath(out_path, ROOT)} ({size_mb:.1f} MB) in {elapsed:.2f}s")
    return dict(np.load(out_path, allow_pickle=False))


if __name__ == "__main__":
    data = build_cache(source="tactical", force=True)
    has_ball = ~np.isnan(data["ball_x"])
    print(f"[cache_atd] Frames={len(has_ball)}, slots={data['player_x'].shape[1]}, ball vis={100 * has_ball.mean():.1f}%")

