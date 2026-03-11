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

*To be filled after running `scripts/estimate_duplicates.py`.*

### Per-Species Duplication

<!-- Paste output table here after running -->

### Overall Summary

<!-- Paste overall stats here -->

### Sensitivity Analysis

<!-- Compare results at threshold 6, 8, 10 if run -->

---

## Next Steps

Depending on results:

1. **If duplication rate is low (< 15%)** — duplicates are not a major concern; proceed with current dataset and note the caveat
2. **If duplication rate is moderate (15–40%)** — consider deduplicating before training; compare model performance with and without dedup
3. **If duplication rate is high (> 40%)** — deduplication is strongly recommended before drawing conclusions about model performance

For actual deduplication (not just estimation):
- Use the same pHash clustering to select one representative crop per cluster
- Tiebreaker options (since no confidence scores exist): largest bounding box area, most centered in tile, or random selection
- Regenerate train/val/test splits from the deduplicated dataset

### Relevant Files

| Artifact | Path |
|----------|------|
| Duplication estimate script | `scripts/estimate_duplicates.py` |
| Duplication report (CSV) | `data_exploration/duplication_report.csv` |
| Crop dataset | `data/crops/` |
| Data lineage documentation | `data/DATA_LINEAGE.md` |
