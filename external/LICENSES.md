# External training and calibration data — sources and licences

These datasets cover **other matches only**. They are used solely to train/calibrate external models and ensemble weights:
- `tools/train_super_hybrid.py`: IDSSE (noise v3) + SkillCorner Open Data (Variant B `live/dead` only).
- `tools/sweep_ensemble_weights.py`: Metrica Sports sample games (harmonized labels + noise v3).

No data about the provided FC Barcelona vs Athletic Club match comes from here, and the submission is regenerated from `./data` plus frozen `models/` weights (`tcn_variant_{a,b}.pt`, `hgb_variant_{a,b}.pkl`, `nova_pathcrf_best_team.pt`, `nova_pathcrf_args.json`) and `config/ensemble_weights.json` alone. `MANIFEST.json` records the SHA-256 of IDSSE and Metrica source files (`python tools/fetch_external.py`).

## 1. IDSSE — integrated spatiotemporal and event data (Bundesliga)

- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and
  event data in elite soccer.* Scientific Data 12, 195.
- figshare: https://doi.org/10.6084/m9.figshare.28196177
- Data owner: Deutsche Fußball Liga (DFL) / Sportec Solutions.
- **Licence: Creative Commons Attribution 4.0 (CC BY 4.0).**
- Used: 7 matches, TRACAB optical tracking at 25 fps (2-D positions only). Per-frame `BallStatus`
  (alive/dead) and `BallPossession` (team) are used as labels. Simulated automatic-tracking noise
  (`noise v3`) is applied during model training (`tools/train_super_hybrid.py`).

## 2. Metrica Sports sample data

- https://github.com/metrica-sports/sample-data
- **Licence: none stated.** The README asks users to "be responsible with the use of this data" and,
  for any public use, to "acknowledge the source". We acknowledge Metrica Sports as the source.
- Used: 3 anonymised matches, 25 fps tracking synchronised with event data. Used for ensemble weight
  selection and nested LOMO validation (`tools/sweep_ensemble_weights.py`, never trained on by
  `PerObjectTCN`, `HGB`, or `Nova PathCRF`). Possession / dead-ball labels are derived **algorithmically**
  from the events (`PASS`, `RECOVERY`, `BALL LOST`, `BALL OUT`, `SET PIECE`, …) by `src/external/metrica.py`
  and harmonised via `src/external/harmonize.py` following `CHALLENGE.md`'s controlled-possession rule.

## 3. SkillCorner Open Data (A-League broadcast tracking)

- https://github.com/SkillCorner/opendata (commit `4340d27`)
- **Licence: MIT Licence.**
- Used: 20 complete A-League 2024/25 matches recorded at 10 fps (`_tracking_extrapolated.jsonl`, upsampled linearly to 25 fps with 20 s slot-identity shuffling). Used exclusively in Variant B training (`tools/train_super_hybrid.py`) for `live/dead` supervision (`poss_weight = 0.0`, live spans taken from `_phases_of_play.csv`; SkillCorner possession labels are not used due to differing possession definitions) and as an external real-tracker validation benchmark.

## 4. PathCRF & Nova Solutions Fine-Tuned Weights (`src/models/nova_pathcrf.py`)

- **Base Architecture:** Kim, H., Lee, K., Seo, S., Ko, S.-K., Yoon, J., & Park, C. (KDD 2026). *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories.* Code and Sportec checkpoint `saved/140` (`cv_fold_idx = 6`, 6 IDSSE Bundesliga matches): https://github.com/hyunsungkim-ds/pathcrf (commit `1b7c6f8`). **Licence: Mozilla Public License 2.0 (MPL-2.0).**
- **Fine-Tuned Checkpoint & Public HF Verification:** Nova Solutions (`Youssef16Ahmed/nova-pathcrf-finetuned`, `round3/final/model/state_dict_best_team.pt` & `round3/final/args.json`): https://huggingface.co/Youssef16Ahmed/nova-pathcrf-finetuned.
  - **Public Repository Metadata (verified unauthenticated `HTTP 200`):** `private: false`, `gated: false`, HF commit SHA `c86163f1c40c5f937879a9f9ed2b1b5031212ccb` (`createdAt: 2026-10-17T10:28:38Z`, `lastModified: 2026-10-17T13:31:02Z`).
  - **Licence Declaration:** `license: mpl-2.0` explicitly declared in both Hugging Face Hub `cardData` metadata and repository `README.md` YAML frontmatter, inheriting **Mozilla Public License 2.0 (MPL-2.0)** from upstream `hyunsungkim-ds/pathcrf`.
  - **Training Data Provenance (`README.md` & `round3/FINAL.json`):** Fine-tuned exclusively on **SkillCorner Open Data** (`https://github.com/SkillCorner/opendata`, MIT) across a 14 train / 1 validation (`2017461`, `B_acc_pathcrf = 0.7365`) / 5 test (`2010085`, `2016236`, `2006229`, `1959846`, `1899585`) A-League split with `ball_feature = false`.
  - **Bundled File Checksums:**
    - `models/nova_pathcrf_best_team.pt`: `719cb853446431948a8879d97ea98019357f04bc32801e241740b32b77aa5586` (`19,325,599` bytes)
    - `models/nova_pathcrf_args.json`: `71683827431d24ddc71292b319b7c60d59fd7778d6ea27ed82df60cce355ac85` (`1,256` bytes)
- **Integration Scope:** Adapted into standalone MPL-2.0-headed module `src/models/nova_pathcrf.py` and ensembled with `PerObjectTCN` and `HistGradientBoostingClassifier` using weights calibrated on unseen Metrica sample games (`tools/sweep_ensemble_weights.py`). Zero `BAR-ATH` labels or data were used in pretraining, fine-tuning, or weight selection.
