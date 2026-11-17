# Ensemble Weight Calibration (`TCN + HGB + Nova PathCRF`)

Evaluated on external matches (**Metrica Sample Games 1, 2, 3** with harmonized controlled-possession labels via `src/external/harmonize.py` and synthetic `noise v3` matching ~41% missing ball and 20s slot swaps; `seed=0`). None of these 3 matches were used to train `PerObjectTCN`, `HGB`, or `Youssef16Ahmed/nova-pathcrf-finetuned`. Because `config/ensemble_weights.json` was selected on all 3 matches (`0.7502` in-grid, `+0.0287` selection optimism), the **strict nested 3-fold LOMO** score (**`0.7215`**, **`+0.0295`** over the `0.6920` 2-model baseline) measures unbiased out-of-sample weight generalization.

## 1. Selected Configuration (`config/ensemble_weights.json`)

- **Blend Mode:** `visibility_gated` (`readout = crf_src`, `switch_delay_s = 1.0`, `stride_seconds = 10.0`)
- **`P(home | live)` (Ball Visible):** `TCN_A = 0.4875`, `HGB_A = 0.2625`, `Nova_PathCRF = 0.2500`
- **`P(home | live)` (Ball Missing):** `TCN_A = 0.5850`, `HGB_A = 0.3150`, `Nova_PathCRF = 0.1000`
- **`P(live)`:** `TCN_B = 0.3800`, `HGB_B = 0.5700`, `Nova_PathCRF_Macro = 0.0500`
- **Viterbi Penalties:** `c_team = 5.0`, `c_dead = 30.0`

## 2. Strict Nested 3-Fold Leave-One-Match-Out

Each fold runs the full weight grid selection on 2 Metrica matches and evaluates blindly on the 3rd held-out match:

| Held-Out Match | 2-Model Baseline `spellF1` | Nested LOMO `spellF1` | Nested LOMO `macroF1` | `spellF1` Gain | Fold Selection `(readout, delay, c_team/c_dead)` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `metrica_1` | 0.6499 | **0.6941** | 0.8400 | **+0.0442** | `crf_src`, `0.0s`, `(5/25)` |
| `metrica_2` | 0.6811 | **0.7011** | 0.8309 | **+0.0200** | `crf_src`, `1.0s`, `(4/25)` |
| `metrica_3` | 0.7450 | **0.7692** | 0.8769 | **+0.0243** | `crf_src`, `0.0s`, `(5/30)` |
| **Mean (3 Held-Out Folds)** | **0.6920** | **0.7215** | **0.8492** | **+0.0295** | Mean `f1_D = 0.8327` |
