# Duplication Assessment

Estimating the rate of duplicate bird crops in the classification dataset.

---

## Background

The classification crop dataset (`data/crops/`) was produced by cropping individual birds from overlapping 512x512 tiles (see `data/DATA_LINEAGE.md` for full lineage). Because tiles overlap spatially, the same physical bird can appear in multiple tiles, producing multiple nearly-identical crops. Hank Arnold confirmed this is by design:

> "The program that exports these tiles is designed to export every single annotated bird at least once."
> "In crowded colonies, many birds are exported more than once, but in a different part of the image each time."

No deduplication was performed during cropping (`data/saahil_code/crop_and_pad.py`).

### Why this matters

- **Inflated class counts** — the 98k crops likely overcount the number of unique birds
- **Biased training** — birds near tile boundaries are seen more often than interior birds, skewing what the model learns
- **Not true augmentation** — unlike intentional augmentation (flips, jitter), these are near-identical copies that don't add meaningful variety
- **Distribution distortion** — species in denser colonies (more tile overlaps) may be disproportionately inflated

### PI Guidance

Dr. Barman's guidance on deduplication:
- If two bounding boxes have **IoU > 0.8**, they represent the same bird
- When duplicates are identified, keep the detection with the **highest confidence score**
- However, the current dataset is **ground-truth annotations** (from Hank's annotation app, not Co-DETR predictions), so no confidence scores are available
- The detection pipeline does perform deduplication for counting (using lat/lon reprojection), but the classification training export was **not deduplicated**

---

## Approach: Perceptual Hash Duplication Estimate

Since we lack global tile coordinates to compute spatial IoU across tiles, we use **perceptual hashing** to estimate duplication from the crop images directly.

### Method

For each species folder independently:
1. Compute a perceptual hash (pHash) of every crop image
2. Cluster crops whose hashes are within a hamming distance threshold
3. Each cluster represents one likely unique bird
4. Crops beyond the first in each cluster are counted as duplicates

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Hash algorithm** | pHash (perceptual hash) | DCT-based, robust to minor crop/resize differences; better than aHash/dHash for near-duplicate detection |
| **Hash size** | 8x8 (default) | Produces a 64-bit hash per image |
| **Hamming distance threshold** | 8 | Two crops are considered the same bird if their hashes differ by ≤ 8 bits out of 64 (~12.5% tolerance) — permissive enough for slightly shifted crops from overlapping tiles, strict enough to avoid merging genuinely different birds |
| **Clustering** | Greedy single-pass | Each crop is compared to existing cluster representatives; assigned to the first match or starts a new cluster |
| **Scope** | Within each species folder | Only compares crops of the same species (a ROTE crop is never compared to a LAGU crop) |
| **Parallelism** | 16 workers (multiprocessing) | Hashing is I/O-bound; parallelizes image loading + hashing across CPU cores |
| **Input** | `data/crops/<SPECIES>/*.jpg` | ~98k crops across 32 species folders |
| **Output** | `data_exploration/duplication_report.csv` | Per-species and overall duplication stats |

### Metrics Reported (per species)

| Metric | Description |
|--------|-------------|
| `total_crops` | Number of crop images in the species folder |
| `unique_birds_approx` | Estimated unique birds (number of hash clusters) |
| `duplicate_crops` | Total crops minus unique birds |
| `duplication_rate_pct` | Duplicate crops as percentage of total |
| `max_cluster_size` | Largest group of crops identified as the same bird |
| `birds_with_duplicates` | Number of unique birds that have more than one crop |

### How to Run

```bash
# Default settings (threshold=8, 16 workers)
time python scripts/estimate_duplicates.py

# Stricter matching (fewer duplicates detected)
python scripts/estimate_duplicates.py --threshold 6

# More permissive matching (more duplicates detected)
python scripts/estimate_duplicates.py --threshold 10

# Custom workers
python scripts/estimate_duplicates.py --workers 24
```

### Limitations

- **Greedy clustering is order-dependent** — not globally optimal, but sufficient for estimating duplication rates
- **Threshold sensitivity** — threshold 8 is a starting point; results should be compared across thresholds (6, 8, 10) to assess sensitivity
- **False merges possible** — two genuinely different but visually similar birds of the same species could be falsely clustered (unlikely given 224x224 crops include background context)
- **False misses possible** — if a bird appears at very different scales or orientations across tiles, pHash may not match them (unlikely since crops are all resized to 224x224)

---

## Results

Run date: 2026-03-11 | Instance: Lambda gpu_1x_a10 (us-west-1) | Runtime: 36m 39s

### Overall Summary

| Metric | Value |
|--------|-------|
| Total crops | 97,996 |
| Estimated unique birds | ~58,084 |
| Duplicate crops | ~39,912 |
| **Overall duplication rate** | **40.7%** |

Nearly **2 out of every 5 crops** in the dataset are duplicates of another crop.

### Per-Species Duplication

| Species | Total Crops | Unique Birds (approx) | Duplicate Crops | Dup Rate (%) | Max Cluster | Birds w/ Dups |
|---------|------------:|----------------------:|----------------:|-------------:|------------:|--------------:|
| MTRN | 62,064 | 30,786 | 31,278 | 50.4 | 35 | 16,755 |
| LAGU | 11,205 | 9,227 | 1,978 | 17.7 | 7 | 1,530 |
| ROTE | 6,325 | 3,929 | 2,396 | 37.9 | 10 | 1,658 |
| BRPE | 3,487 | 3,116 | 371 | 10.6 | 4 | 336 |
| WHIB | 2,403 | 1,579 | 824 | 34.3 | 7 | 587 |
| BLSK | 1,675 | 1,054 | 621 | 37.1 | 13 | 364 |
| REEG | 1,581 | 1,264 | 317 | 20.1 | 6 | 251 |
| GBHE | 1,525 | 1,272 | 253 | 16.6 | 6 | 213 |
| TRHE | 1,250 | 1,027 | 223 | 17.8 | 4 | 193 |
| CAEG | 1,123 | 636 | 487 | 43.4 | 12 | 296 |
| BCNH | 1,119 | 833 | 286 | 25.6 | 6 | 223 |
| ROSP | 1,003 | 756 | 247 | 24.6 | 6 | 187 |
| SATE | 653 | 371 | 282 | 43.2 | 4 | 181 |
| GREG | 545 | 502 | 43 | 7.9 | 6 | 34 |
| MEGRT | 467 | 434 | 33 | 7.1 | 3 | 31 |
| BRPEC | 411 | 380 | 31 | 7.5 | 2 | 31 |
| REEGWM | 320 | 241 | 79 | 24.7 | 6 | 62 |
| OTHR | 252 | 215 | 37 | 14.7 | 4 | 33 |
| SNEG | 208 | 151 | 57 | 27.4 | 4 | 48 |
| WHIBC | 151 | 97 | 54 | 35.8 | 3 | 44 |
| GREGC | 92 | 88 | 4 | 4.3 | 2 | 4 |
| TCHE | 55 | 53 | 2 | 3.6 | 2 | 2 |
| RUTU | 37 | 32 | 5 | 13.5 | 2 | 5 |
| AWPE | 20 | 19 | 1 | 5.0 | 2 | 1 |
| WHIBJ | 6 | 5 | 1 | 16.7 | 2 | 1 |
| BNST | 5 | 5 | 0 | 0.0 | 1 | 0 |
| AMAV | 4 | 3 | 1 | 25.0 | 2 | 1 |
| AMOY | 2 | 2 | 0 | 0.0 | 1 | 0 |
| BEKI | 2 | 2 | 0 | 0.0 | 1 | 0 |
| DCCO | 2 | 1 | 1 | 50.0 | 2 | 1 |
| RBGU | 2 | 2 | 0 | 0.0 | 1 | 0 |
| OSPR | 1 | 1 | 0 | 0.0 | 1 | 0 |
| TUVU | 1 | 1 | 0 | 0.0 | 1 | 0 |

---

## Interpretation

### The duplication rate is high — deduplication is recommended

At **40.7% overall**, the dataset falls squarely in the "high duplication" category. This is not surprising given Hank's explicit statement that the export prioritizes recall and intentionally includes birds multiple times across overlapping tiles.

### Duplication is not uniform across species

The duplication rate varies dramatically by species, ranging from 0% to 50.4%:

- **Heavily duplicated (> 35%)**: MTRN (50.4%), CAEG (43.4%), SATE (43.2%), ROTE (37.9%), BLSK (37.1%), WHIBC (35.8%), WHIB (34.3%)
- **Moderately duplicated (15–35%)**: SNEG (27.4%), BCNH (25.6%), REEGWM (24.7%), ROSP (24.6%), REEG (20.1%), TRHE (17.8%), LAGU (17.7%), GBHE (16.6%)
- **Low duplication (< 15%)**: BRPE (10.6%), GREG (7.9%), BRPEC (7.5%), MEGRT (7.1%)

This non-uniformity is likely driven by colony density — dense nesting colonies (terns, egrets) have more birds near tile boundaries, producing more overlapping detections.

### MTRN dominance is even more extreme than it appeared

MTRN goes from 62,064 crops to ~30,786 unique birds — still dominant, but the raw count was inflated by ~2x. Meanwhile species like BRPE (10.6% dup rate) are less inflated. This means the true class imbalance is **different from what the raw crop counts suggest**. After deduplication, MTRN's share of the dataset would decrease relative to less-duplicated species.

### Max cluster sizes confirm tile-overlap duplication pattern

The largest duplicate clusters (MTRN max=35, BLSK max=13, CAEG max=12) suggest some birds appear in up to 35 overlapping tiles. This is consistent with birds at the intersection of many tiles in dense colony centers — exactly the pattern expected from overlapping 512x512 tiling of dense rookery imagery.

### Impact on training

The current dataset effectively gives tile-boundary birds **up to 35x more training weight** than interior birds. With `WeightedRandomSampler` applied on top (which weights by 1/class_count using the inflated counts), the actual per-bird sampling probabilities are distorted in ways that are difficult to reason about. Deduplication would give a cleaner foundation for any subsequent balancing strategy.

---

## Next Steps

Given the **40.7% duplication rate**, deduplication is strongly recommended before further training experiments.

### Recommended approach

1. **Deduplicate using pHash clustering** — reuse the same method from estimation, but output a deduplicated crop list (one representative per cluster)
2. **Tiebreaker for which crop to keep** (since no confidence scores exist):
   - Largest bounding box area (from `classification_original_training.json`) — larger bbox = more of the bird visible
   - Most centered crop in its tile — less edge truncation
   - Or simply: first crop encountered (arbitrary but deterministic)
3. **Regenerate metadata and splits** from the deduplicated set
4. **Compare model performance** on deduplicated vs original dataset to quantify the effect

### Open question

Whether to deduplicate before or after class consolidation (grouping rare species into OTHERS). Deduplicating first gives truer counts for deciding the grouping threshold.

### Relevant Files

| Artifact | Path |
|----------|------|
| Duplication estimate script | `scripts/estimate_duplicates.py` |
| Duplication report (CSV) | `data_exploration/duplication_report.csv` |
| Crop dataset | `data/crops/` |
| Data lineage documentation | `data/DATA_LINEAGE.md` |
