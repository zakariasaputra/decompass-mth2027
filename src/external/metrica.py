"""Metrica Sports sample games -> MatchData with algorithmically derived possession labels.

Source: https://github.com/metrica-sports/sample-data (other matches; acknowledged per README).

Labels are derived by a fixed algorithm from Metrica's event data (human-produced data about
OTHER games, which RULES.md allows):

* Dead ball (D): for every SET PIECE except the opening kick-off of a period, play was dead from the
  end of the previous event (the moment the ball crossed the line, the foul, the goal, ...) until
  the set piece is taken.
* Possession (H/A) during live play: the team of the most recent on-ball event (PASS, CARRY, SHOT,
  SET PIECE, RECOVERY, BALL LOST, BALL OUT). Possession switches at the start frame of the first
  on-ball event by the other team (typically its RECOVERY). CHALLENGE / CARD / FAULT RECEIVED do
  not change possession.
* Graded blocks: from the opening kick-off of each period to the end of its last event.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm  # noqa: E402
from src.matchdata import MatchData  # noqa: E402

MET_DIR = os.path.join(ROOT, "external", "metrica")
CACHE_DIR = os.path.join(ROOT, "external", "cache")
ON_BALL = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT"}
PITCH = (105.0, 68.0)


def derive_labels(events, n_frames):
    """events: DataFrame with team (0/1), type, subtype, period, start, end (0-based frame indices)."""
    labels = np.full(n_frames, pm.X, dtype=np.int8)
    blocks = []
    for period, ev in events.groupby("period", sort=True):
        ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
        ko = ev[(ev["type"] == "SET PIECE") & (ev["subtype"].str.contains("KICK OFF", na=False))]
        start = int(ko["start"].iloc[0]) if len(ko) else int(ev["start"].iloc[0])
        last = int(min(n_frames - 1, max(ev["end"].max(), ev["start"].max())))
        blocks.append((int(period), start, last))

        # Possession: last on-ball team, switching at the first on-ball event of the other team
        team_track = np.full(last - start + 1, -1, dtype=np.int8)
        cur, cur_from = -1, start
        for _, row in ev.iterrows():
            if row["type"] not in ON_BALL or row["team"] not in (0, 1):
                continue
            f = int(np.clip(row["start"], start, last))
            if row["team"] != cur:
                if cur in (0, 1):
                    team_track[cur_from - start:f - start] = cur
                cur, cur_from = int(row["team"]), f
        if cur in (0, 1):
            team_track[cur_from - start:] = cur
        # Before the first on-ball event, use the kick-off team
        first_known = np.flatnonzero(team_track >= 0)
        if len(first_known):
            team_track[:first_known[0]] = team_track[first_known[0]]
        seg = np.where(team_track == 0, pm.H, pm.A).astype(np.int8)

        # Dead ball before every set piece except the period's opening kick-off
        for i, row in ev.iterrows():
            if row["type"] != "SET PIECE" or int(row["start"]) <= start:
                continue
            prev = ev.iloc[:i]
            prev = prev[prev["start"] < row["start"]]
            if not len(prev):
                continue
            d0 = int(np.clip(prev["end"].max(), start, last))
            d1 = int(np.clip(row["start"], start, last))
            if d1 > d0:
                seg[d0 - start:d1 - start] = pm.D
        labels[start:last + 1] = seg
    return labels, blocks


def _load_csv_game(game_no):
    gdir = os.path.join(MET_DIR, f"Sample_Game_{game_no}")
    px, py, teams = [], [], []
    ball = None
    n = None
    for team_code, side in ((0, "Home"), (1, "Away")):
        path = os.path.join(gdir, f"Sample_Game_{game_no}_RawTrackingData_{side}_Team.csv")
        df = pd.read_csv(path, skiprows=2)
        cols = list(df.columns)
        n = len(df)
        for j, c in enumerate(cols):
            if str(c).startswith("Player"):
                px.append(df.iloc[:, j].to_numpy(dtype=np.float32))
                py.append(df.iloc[:, j + 1].to_numpy(dtype=np.float32))
                teams.append(team_code)
            if c == "Ball" and ball is None:
                ball = (df.iloc[:, j].to_numpy(dtype=np.float32), df.iloc[:, j + 1].to_numpy(dtype=np.float32))
        frames = df["Frame"].to_numpy()
        assert frames[0] == 1 and np.all(np.diff(frames) == 1), "unexpected frame numbering"

    ev = pd.read_csv(os.path.join(gdir, f"Sample_Game_{game_no}_RawEventsData.csv"))
    events = pd.DataFrame({
        "team": ev["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
        "type": ev["Type"].astype(str).str.upper(),
        "subtype": ev["Subtype"].astype(str).str.upper(),
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int) - 1,
        "end": np.maximum(ev["End Frame"].astype(int), ev["Start Frame"].astype(int)) - 1,
    })
    return np.stack(px, 1), np.stack(py, 1), np.array(teams), ball[0], ball[1], events, n


def _load_epts_game3():
    """Sample Game 3 (EPTS tracking + JSON events), parsed via kloppy for the tracking."""
    from kloppy import metrica

    gdir = os.path.join(MET_DIR, "Sample_Game_3")
    ds = metrica.load_tracking_epts(
        meta_data=os.path.join(gdir, "Sample_Game_3_metadata.xml"),
        raw_data=os.path.join(gdir, "Sample_Game_3_tracking.txt"),
    )
    home, away = ds.metadata.teams
    players = [p for p in home.players] + [p for p in away.players]
    teams = np.array([0] * len(home.players) + [1] * len(away.players))
    n = len(ds.records)
    first_frame = ds.records[0].frame_id
    px = np.full((n, len(players)), np.nan, dtype=np.float32)
    py = np.full((n, len(players)), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    col = {p.player_id: j for j, p in enumerate(players)}
    for i, fr in enumerate(ds.records):
        for p, pdata in fr.players_data.items():
            if pdata.coordinates is not None and p.player_id in col:
                px[i, col[p.player_id]] = pdata.coordinates.x
                py[i, col[p.player_id]] = pdata.coordinates.y
        if fr.ball_coordinates is not None:
            bx[i], by[i] = fr.ball_coordinates.x, fr.ball_coordinates.y

    with open(os.path.join(gdir, "Sample_Game_3_events.json"), encoding="utf-8") as f:
        raw = json.load(f)["data"]
    team_ids = {home.team_id: 0, away.team_id: 1}
    rows = []
    for e in raw:
        st = (e.get("subtypes") or {})
        if isinstance(st, list):
            st = st[0] if st else {}
        rows.append({
            "team": team_ids.get((e.get("team") or {}).get("id"), -1),
            "type": str(e["type"]["name"]).upper(),
            "subtype": str(st.get("name", "")).upper(),
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]) - first_frame,
            "end": max(int(e["end"]["frame"]), int(e["start"]["frame"])) - first_frame,
        })
    return px, py, teams, bx, by, pd.DataFrame(rows), n


def _load_metrica_game_raw(game_no, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"metrica_{game_no}.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
        return MatchData(f"metrica_{game_no}", 25.0, PITCH, z["player_x"], z["player_y"], z["team_of_slot"],
                         z["ball_x"], z["ball_y"], blocks, z["labels"])

    if game_no in (1, 2):
        px, py, teams, bx, by, events, n = _load_csv_game(game_no)
    else:
        px, py, teams, bx, by, events, n = _load_epts_game3()
    labels, blocks = derive_labels(events, n)
    np.savez_compressed(cache, player_x=px, player_y=py, team_of_slot=teams, ball_x=bx, ball_y=by,
                        blocks=np.array(blocks), labels=labels)
    return MatchData(f"metrica_{game_no}", 25.0, PITCH, px, py, teams, bx, by, blocks, labels)


def load_metrica_game(game_no, use_cache=True, harmonize=True):
    """Load a match; with ``harmonize`` (default) labels follow CHALLENGE.md's controlled-possession rule."""
    m = _load_metrica_game_raw(game_no, use_cache=use_cache)
    if harmonize:
        from src.external.harmonize import harmonize_labels

        m.labels = harmonize_labels(m.labels, m.blocks, m.fps)
    return m


if __name__ == "__main__":
    for g in (1, 2, 3):
        m = load_metrica_game(g, use_cache=False)
        print(m.name, m.n_frames, m.blocks, "ball vis %.1f%%" % (100 * np.mean(~np.isnan(m.ball_x))))

