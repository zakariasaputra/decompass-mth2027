# Public 5-Minute Verification (`BAR-ATH`)

Report-only pipeline check against the public 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, frames `11185..18684`, 17 GT possessions of mean 14.2 s). No model weights, ensemble weights, or decoder parameters in the submitted pipeline were trained or selected on this window (`RULES.md`).

- **Submission Artifact:** `submission/Decompass.csv` (`SHA-256: 61d792693de164b87f9b810df03c3699ccb15ea4d9437900ae735c1242b973fb`)
- **Format & Alignment Checks:** Row count `199,319` (`OK`), `X` strictly on `40,462` ungraded frames (`OK`), zero `X` inside graded play (`OK`), `H`/`A` polarity aligned (`OK`).
- **Full-Match Summary (GT-free):** `258` live spells (mean 13.7 s), dead ball 44.4% of graded frames (`H: 54,924`, `A: 33,394`, `D: 70,539`, `X: 40,462`).

| Metric (Public 5-Minute Window) | `Decompass.csv` (`3-Model`) | `baseline_smooth50` | `baseline_smooth125` |
| :--- | :---: | :---: | :---: |
| `spellF1` (mean IoU 0.3–0.7) | **0.6270** | 0.3256 | 0.4242 |
| `spellF1` `@0.3 / @0.4 / @0.5 / @0.6 / @0.7` | 0.7568 / 0.7027 / 0.6486 / 0.5405 / 0.4865 | — | — |
| `macroF1` | **0.7910** | 0.6207 | 0.5812 |
| Per-Class F1 (`H / A / D`) | 0.8769 / 0.8682 / 0.6280 | — | — |
| Transitions F1 `@25f` (`lat_f`) | 0.4091 (`+0.8f`) | — | — |
| Predicted Spells (`mean duration`, GT: `17`, `14.2 s`) | 20 (`13.4 s`) | 26 (`10.8 s`) | 16 (`18.5 s`) |
