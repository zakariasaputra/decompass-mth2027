"""Common match representation shared by the provided match and external calibration matches."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchData:
    name: str
    fps: float
    pitch: tuple                  # (length_m, width_m)
    player_x: np.ndarray          # (T, S) normalised 0..1 (x along length), NaN when untracked
    player_y: np.ndarray          # (T, S) normalised 0..1
    team_of_slot: np.ndarray      # (S,) 0 = home, 1 = away, -1 = other
    ball_x: np.ndarray            # (T,) normalised, NaN when missing
    ball_y: np.ndarray
    blocks: list                  # [(half, start, last)] graded frame blocks (inclusive last)
    labels: np.ndarray = None     # (T,) possession_metrics codes (H/A/D, X outside blocks); external only
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball_x)

