# Decompass — Technical Document

**Team:** Decompass | **Challenge:** More than a Hack 2027 (Phase 1: Ball Possession)  
**Code Repository:** https://github.com/zakariasaputra/decompass-mth2027.git  
**Output SHA-256:** `61d792693de164b87f9b810df03c3699ccb15ea4d9437900ae735c1242b973fb`

## 1. Approach

Our pipeline (`run_pipeline.py`) predicts frame-level ball possession from 2D player/ball tracking and kickoff tags without reading raw video pixels. It combines cross-camera ball recovery, three complementary model families trained exclusively on external matches, a rule-based pre-kickoff gate, and a 3-state Viterbi decoder aligned with spell-level evaluation (`spellF1`):

```
Tactical ATD ──┬──► Ball-free PathCRF at 5 fps (src/models/nova_pathcrf.py)
               │      -> P_home(crf_src, 1.0 s delay) & P_live(macro)
Broadcast ATD ─┼──► Per-half sync & plausibility filter (src/broadcast_align.py)
Homography ────┘      -> +2,943 recovered graded ball frames (vis 59.3% -> 61.2%)
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
  PerObjectTCN (A/B)   Asymmetric HGB (A/B)     Nova PathCRF
         └──────────┬──────────┴──────────┬──────────┘
                    ▼                     ▼
         P(live) log-odds       Ball-visibility-gated P(home | live)
       (0.38 TCN_B + 0.57      vis:     0.4875 TCN_A + 0.2625 HGB_A + 0.2500 PathCRF
        HGB_B + 0.05 PathCRF)  missing: 0.5850 TCN_A + 0.3150 HGB_A + 0.1000 PathCRF
                    │                     │
                    ▼                     │
       Kickoff gate (Kick Off Start tag)  │
                    └──────────┬──────────┘
                               ▼
       3-state Viterbi per half (c_team = 5.0, c_dead = 30.0)
                               ▼
       199,319 frames in {H, A, D, X} (submission/Decompass.csv)
```

Three design choices drive this architecture:

1. **Combining ball-dependent encoders with ball-free graph inference:** Even after multi-camera recovery, the ball is missing on 38.8% of graded frames, and deflections or close challenges often place it near an opponent. Ball-dependent `PerObjectTCN` (12.5 fps permutation-equivariant 2D-Conv over 10.2 s receptive fields) and `HistGradientBoostingClassifier` (162 multi-window features spanning -8 s to +8 s) learn geometric ball control and occlusion persistence. Ensembling them with ball-free `Youssef16Ahmed/nova-pathcrf-finetuned` (`SetTransformer + SAB + DynamicSparseCRF`, `ball_feature = false`) stabilizes contested passages from player kinematics alone.
2. **Ball-visibility-gated log-odds blending:** On frames with an observed ball, weighting 25.0% `Nova PathCRF` alongside 48.75% `TCN_A` and 26.25% `HGB_A` stabilizes duels and rapid passes. During ball gaps, shifting weight back to `TCN_A` (58.5%) and `HGB_A` (31.5%) preserves longer temporal memory across occlusions.
3. **Transition-penalized Viterbi decoding:** Framewise classification over-segments long possessions. Decoding `P(live)` and `P(home | live)` jointly with transition costs `c_team = 5.0` and `c_dead = 30.0` removes flicker and outputs 258 whole-match live spells (mean 13.7 s, 44.4% dead frames).

## 2. Assets Used and Dropped

| Asset | Treatment | Details |
| :--- | :--- | :--- |
| Tactical ATD | **Used** | Primary metric coordinates across 46 slots and ball; step speed capped at 12.0 m/s to reject slot-ID swaps (`src/team_signals.py`, `src/models/nova_pathcrf.py`). |
| Broadcast ATD & Homography | **Used** | Algorithmically aligned per half (`src/broadcast_align.py`) using Otsu constellation (< 1.21 m), Tukey reprojection (< 0.67), and log-Otsu ball plausibility (< 4.97 m). Recovers 2,943 missing graded ball frames (changes 1,118 labels relative to raw tactical ball). |
| Smart Tagging XML | **Selective** | `SET PIECES: Kick Off (Start)` windows anchor half kickoffs (`src/features.py:detect_kickoff_frame`). Tactical phase tags (`BUILD UP`, etc.) were dropped after external evaluation. |
| Tactical Homography & MP4 Video | **Dropped** | Not needed because tactical ATD is already in metric coordinates and zero video pixels are read. |
| Public 5-minute Ground Truth | **Check only** | Used for pipeline verification only (`reports/public_sanity_check.md`); no model weights, ensemble weights, or decoder parameters in the submitted pipeline were trained or selected on this window. |

## 3. Training Data

All learned models and ensemble weights were trained or calibrated strictly on external football datasets (`external/LICENSES.md`):

| Source & Licence | Scope & Label Origin | Role in Pipeline |
| :--- | :--- | :--- |
| **IDSSE Bundesliga** (`doi:10.6084/m9.figshare.28196177`, **CC BY 4.0**) | 7 full matches, 25 fps optical tracking with DFL `BallStatus` and `BallPossession` labels. | Trains `TCN_A/B` and `HGB_A/B` (`tools/train_super_hybrid.py`) under synthetic missing-ball/slot-swap noise (`noise v3`). Base PathCRF `saved/140` also pretrained on 6 DFL matches. |
| **SkillCorner Open Data** (`github.com/SkillCorner/opendata` @ `4340d27`, **MIT**) | 20 A-League matches, 10 fps broadcast tracking upsampled to 25 fps with `_phases_of_play.csv` live spans. | Trains live/dead head in `TCN_B` and `HGB_B` (`poss_weight = 0.0`). Sole fine-tuning dataset (14 train / 1 val / 5 test, no ball features) for `Youssef16Ahmed/nova-pathcrf-finetuned` (HF commit `c86163f1`, **MPL-2.0**). |
| **Metrica Sample Data** (`github.com/metrica-sports/sample-data`, open source) | 3 matches, 25 fps tracking with event-derived labels harmonized to controlled possession (`src/external/harmonize.py`). | Calibrates `config/ensemble_weights.json` under synthetic `noise v3` (`tools/sweep_ensemble_weights.py`). Never seen during training by any of the three model families. |

To measure ensemble-weight generalization without grid-selection optimism (in-grid 3-match `spellF1` = 0.7502, +0.0582), `tools/sweep_ensemble_weights.py` runs strict nested 3-fold Leave-One-Match-Out (selecting all blend weights and Viterbi penalties on two Metrica matches and testing on the third). Nested LOMO improves mean `spellF1` across all 3 held-out matches from 0.6920 (2-model baseline) to 0.7215 (+0.0295: +0.0442, +0.0200, +0.0243) and `macroF1` from 0.8443 to 0.8492.

## 4. Possession Rules Implemented

- **Controlled team possession (`H` vs `A`):** Frame-level log-odds blend of `PerObjectTCN`, `Asymmetric HGB`, and delayed (1.0 s) ball-free `Nova PathCRF` (`crf_src` forward-backward marginals), switching teams when accumulated evidence overcomes Viterbi cost `c_team = 5.0` nats so transient mid-flight deflections remain absorbed.
- **Dead-ball intervals (`D`):** Un-absorbed live/dead probabilities (`0.38 TCN_B + 0.57 HGB_B + 0.05 PathCRF_Macro`) decoded with boundary penalty `c_dead = 30.0` nats, plus the IFAB Law 8 pre-kickoff gate (`P(live) = 0.0` from half block start until minimum ball-to-center distance within `Kick Off (Start)` tags, or ball within 3.5 m of center in the first 120 s if no tags are supplied; masks frames 11,185–11,187 and 110,808–113,741).
- **Out-of-bounds / ungraded frames (`X`):** Strictly assigned to all 40,462 frames outside graded halves (`games.half_video_blocks`); zero graded frames abstain.

## 5. Limitations

1. **Missing 3D ball height ($z$):** 2D pitch coordinates cannot separate high aerial passes clearing a defender from ground interceptions without touch continuity.
2. **Synthetic-to-real tracker mismatch & public-window regression:** The external +0.0295 nested-LOMO gain for `Nova PathCRF` was measured on clean Metrica tracking degraded synthetically (`noise v3`), whose error distribution may differ from native BAR-ATH automatic tracking. On the 5-minute verification slice (17 GT possessions), `Decompass.csv` scores `spellF1` = 0.6270 and `macroF1` = 0.7910 (`H` 0.8769, `A` 0.8682, `D` 0.6280), which is 0.036 `spellF1` lower than a 2-model baseline (`TCN + HGB`, within the ~0.04 noise band, driven by lower dead-ball F1).
3. **Anonymous slot packing across 10 s windows:** `Nova PathCRF` assumes 11 consistent player identities per team, whereas BAR-ATH has 46 swapping tracklet slots; packing the top 11 visible slots per non-overlapping 10 s window loses boundary context on long tracklet churn.
4. **Short turnovers (< 2 s):** Transition penalties `(5.0, 30.0)` prevent noisy fragmentation but absorb brief legitimate possessions lasting under two seconds.
5. **Unverified multi-camera fusion benefit:** Broadcast ball recovery shifts 1,118 graded frames relative to tactical-only input (+397 `H`, +390 `A`, -787 `D`), and its net effect on `spellF1` could not be verified externally because all external datasets provide only a single camera view. With more time, we would stitch anonymous tracklets across PathCRF windows and validate dual-camera ball fusion on multi-angle benchmarks.

## 6. Reproduction

Tested with Python 3.12.10 on CPU:

```bash
git clone https://github.com/zakariasaputra/decompass-mth2027.git
cd decompass-mth2027
pip install -r requirements-pipeline.txt
# Place the competition BAR-ATH bundle into ./data (or export MTAH_DATA_DIR)

bash run.sh                                       # rebuilds caches, writes submission/Decompass.csv, validates
```

Equivalent verification commands:

```bash
python run_pipeline.py --rebuild-cache --out submission/Decompass.csv
python evaluation/validate_submission.py submission/Decompass.csv
pytest tests/ -v                                  # 8 invariant and GT-quarantine tests
python tools/clean_room_check.py                  # leak check + deterministic 2x cold & 1x warm SHA-256 verification
```

Reproducing from raw `./data` writes 199,319 rows (`H`: 54,924; `A`: 33,394; `D`: 70,539; `X`: 40,462; 258 spells, mean 13.7 s) with exact SHA-256 `61d792693de164b87f9b810df03c3699ccb15ea4d9437900ae735c1242b973fb`.
