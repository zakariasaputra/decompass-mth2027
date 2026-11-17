"""Align external possession labels with the competition's definition of a change of possession.

CHALLENGE.md: a change of possession needs *controlled* possession. A simple touch, a deflection
that does not intercept, or an aerial duel without subsequent control is NOT a change; "in phases
where the ball is contested or loose, possession stays with the last team that had it".

Provider labels (DFL ``BallPossession``, Metrica events) flip on nearly every touch. This module
applies two fixed, pre-registered rules to a label array (H/A/D/X) -- no tuning, no use of the
provided match or its ground truth:

* bridge:   a run of one team shorter than ``MIN_BRIDGE_S`` (3 s, "a player keeps the ball for a
            few seconds") that lies between two runs of the OTHER team is absorbed by them.
* terminal: a run shorter than ``MIN_TERMINAL_S`` (3 s, same "few seconds" rule) that ends in a dead ball is a deflection
            out of play; it is absorbed by the preceding run.

Runs are merged shortest-first until none qualifies.
"""

import numpy as np

import evaluation.possession_metrics as pm

MIN_BRIDGE_S = 3.0
MIN_TERMINAL_S = 3.0


def _live_runs(lab, lo, hi):
    """Maximal runs of identical H/A inside [lo, hi) as [start, stop, label] lists."""
    runs = []
    t = lo
    while t < hi:
        v = lab[t]
        u = t + 1
        while u < hi and lab[u] == v:
            u += 1
        runs.append([t, u, int(v)])
        t = u
    return runs


def _consolidate_interval(lab, lo, hi, min_bridge, min_terminal):
    """Consolidate one maximal non-dead interval [lo, hi) in place."""
    runs = _live_runs(lab, lo, hi)
    while len(runs) > 1:
        best, best_len = None, None
        for i, (s, e, v) in enumerate(runs):
            n = e - s
            is_last = i == len(runs) - 1
            is_first = i == 0
            bridge = (not is_first) and (not is_last) and n < min_bridge
            terminal = is_last and (not is_first) and hi < len(lab) and n < min_terminal and lab[hi] == pm.D
            if (bridge or terminal) and (best_len is None or n < best_len):
                best, best_len = i, n
        if best is None:
            break
        s, e, _ = runs[best]
        prev = runs[best - 1]
        lab[s:e] = prev[2]
        prev[1] = e
        runs.pop(best)
        if best < len(runs) and runs[best][2] == prev[2]:
            prev[1] = runs[best][1]
            runs.pop(best)
    return lab


def harmonize_labels(labels, blocks, fps=25.0, min_bridge_s=MIN_BRIDGE_S, min_terminal_s=MIN_TERMINAL_S):
    """Return a harmonised copy of ``labels`` (H/A/D/X) without touching D or X frames."""
    lab = np.array(labels, dtype=np.int8, copy=True)
    mb, mt = int(round(min_bridge_s * fps)), int(round(min_terminal_s * fps))
    for _, start, last in blocks:
        stop = last + 1
        live = (lab[start:stop] == pm.H) | (lab[start:stop] == pm.A)
        edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            _consolidate_interval(lab, int(s), int(e), mb, mt)
    return lab
