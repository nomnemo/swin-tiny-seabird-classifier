# Session Log

Daily progress notes for the Swin-Tiny seabird classification project.

---

## 2026-03-11

### Environment Setup
- Set up Lambda gpu_1x_a10 instance (us-west-1)
- Loaded crop dataset from S3 → `data/crops/` (took ~46 min)
- Installed dependencies via `requirements.txt` into venv
- Documented data loading times in `data_loading_times.md`

### Documentation
- Created `data/DATA_LINEAGE.md` — full provenance from Hank's orthomosaics → tiled dataset → Saahil's crops → our splits
- Updated `README.md` with setup.sh option + Claude Code CLI install step

### Duplication Analysis
- Built `scripts/estimate_duplicates.py` — perceptual hash (pHash) based duplication estimator with multiprocessing (16 workers)
- **Result**: 97,996 crops → ~58,084 unique birds (**40.7% duplication rate**)
  - MTRN worst at 50.4% (62k → ~31k unique), max cluster size = 35
  - Duplication non-uniform: dense colony species (terns, egrets) much worse than solitary species
- Created `data_exploration/DUPLICATION_ASSESSMENT.md` with full results and interpretation
- **Decision**: NOT deduplicating via pHash — visual similarity is unreliable for identifying same bird. Two different birds can look similar. Would need spatial/coordinate-based dedup from upstream pipeline, which requires new export from Hank.
- Confirmed group-aware splits (`scripts/DataSplitter.py`) already prevent leakage — all tiles from same orthomosaic stay in same split

### Classifier 1 Setup (14-class)
- Updated `MERGE_GROUPS` in `SwinTrainer.py` for correct 14-class grouping:
  - White birds: GREG + SNEG + WHIB + REEGWM
  - Mixed terns: ROTE + MTRN + SATE
  - Don't care: MEGRT + OTHR + OTHERS
- Analyzed impact of `max_per_class=1000` cap with duplication:
  - Large classes fine (cap samples small fraction)
  - Small classes (CAEG, SATE, WHIBC) may include duplicates since cap ≈ unique count
  - WeightedRandomSampler already handles batch-level balancing
- **Bumped `max_per_class` to 3000** — middle ground between 1000 (too restrictive, wastes data) and no cap (too loose, computationally wasteful)

### Runs
- [ ] Training run: 14-class, max_per_class=3000, 30 epochs — *in progress / queued*

### Next Steps
- Run training, evaluate results
- Compare with previous max_per_class=1000 baseline
- Discuss results with PI
