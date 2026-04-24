# Crop-Split Experiment Comparison: Exp1 (11-class) vs Exp2 (10-class Terns Merged)

**Date:** April 2025
**Dataset:** Chester Island 2025 Cropped Bird Dataset
**Model:** Swin-Tiny (swin_tiny_patch4_window7_224)
**Split Method:** Crop-level 80/10/10 (all classes represented in all splits)

---

## 1. Executive Summary

| Metric | Exp1 (11-class) | Exp2 (10-class) | Winner |
|--------|-----------------|-----------------|--------|
| **Test Macro-F1** | **0.825** | 0.819 | Exp1 |
| **Test mAP** | **0.901** | 0.883 | Exp1 |
| **Test Accuracy** | 89.1% | **93.0%** | Exp2 |
| **Val Macro-F1** | 0.806 | **0.822** | Exp2 |
| **Misclassified (test)** | 157 | **81** | Exp2 |
| **Classes** | 11 | 10 | - |

**Key Finding:** Exp1 achieves better macro-F1 and mAP (class-balanced metrics), while Exp2 achieves higher accuracy (dominated by the large TERNS class). The choice depends on whether you prioritize balanced performance across all species or overall prediction accuracy.

---

## 2. Training Configuration

Both experiments used identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Model | swin_tiny_patch4_window7_224 |
| Epochs | 30 |
| Learning Rate | 1e-4 (with 10-epoch warmup) |
| Weight Decay | 0.01 |
| Batch Size | 32 (train), 128 (eval) |
| Max Per Class | 3000 |
| Sampler | WeightedRandomSampler |
| Hardware | 1x NVIDIA A10 |

---

## 3. Dataset Composition

### Exp1: 11-class Baseline

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| ROTEA | 3000 | 600 | 600 | 4200 |
| SATEA | 2200 | 275 | 276 | 2751 |
| BRPEC | 1992 | 249 | 249 | 2490 |
| LAGUA | 1146 | 143 | 144 | 1433 |
| BRPEA | 655 | 82 | 82 | 819 |
| OTHERS | 297 | 37 | 38 | 372 |
| TRHEA | 137 | 17 | 18 | 172 |
| GREGC | 85 | 11 | 11 | 107 |
| GREGA | 84 | 11 | 11 | 106 |
| LWBBA | 55 | 7 | 7 | 69 |
| GBHEC | 48 | 6 | 6 | 60 |
| **Total** | **9699** | **1438** | **1442** | **12579** |

### Exp2: 10-class (Terns Merged)

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| TERNS | 3000 | 600 | 600 | 4200 |
| BRPEC | 1992 | 249 | 249 | 2490 |
| LAGUA | 1146 | 143 | 144 | 1433 |
| BRPEA | 655 | 82 | 82 | 819 |
| OTHERS | 218 | 27 | 28 | 273 |
| TRHEA | 137 | 17 | 18 | 172 |
| GREGC | 85 | 11 | 11 | 107 |
| GREGA | 84 | 11 | 11 | 106 |
| LWBBA | 55 | 7 | 7 | 69 |
| GBHEC | 48 | 6 | 6 | 60 |
| **Total** | **7420** | **1153** | **1156** | **9729** |

**Note:** Exp2 has fewer samples because ROTEA+SATEA+MTRNS+ROTEF+SATEF were merged into TERNS (capped at 3000), reducing total training samples.

---

## 4. Per-Class Performance Comparison (Test Set)

| Class | Exp1 Precision | Exp1 Recall | Exp1 F1 | Exp2 F1 | Difference |
|-------|----------------|-------------|---------|---------|------------|
| BRPEA | 0.919 | 0.829 | 0.872 | 0.887 | +0.015 |
| BRPEC | 0.936 | 0.884 | 0.909 | 0.928 | +0.019 |
| GBHEC | 0.600 | 0.500 | 0.545 | 0.769 | **+0.224** |
| GREGA | 0.833 | 0.909 | 0.870 | 0.833 | -0.037 |
| GREGC | 0.769 | 0.909 | 0.833 | 0.769 | -0.064 |
| LAGUA | 0.957 | 0.931 | 0.944 | 0.923 | -0.021 |
| LWBBA | 0.667 | 0.857 | 0.750 | 0.706 | -0.044 |
| OTHERS | 0.650 | 0.684 | 0.667 | 0.632 | -0.035 |
| ROTEA | 0.948 | 0.883 | 0.915 | - | - |
| SATEA | 0.775 | 0.949 | 0.853 | - | - |
| TERNS | - | - | - | 0.967 | - |
| TRHEA | 0.941 | 0.889 | 0.914 | 0.778 | **-0.136** |

### Key Observations:

1. **GBHEC improved dramatically in Exp2** (+22.4% F1): With fewer classes, the model better distinguishes Great Blue Heron Chicks.

2. **TRHEA degraded in Exp2** (-13.6% F1): Tri-colored Heron Adults are more confused with other classes when terns are merged.

3. **TERNS class performs excellently** (96.7% F1): Merging tern species eliminates the ROTEA↔SATEA confusion present in Exp1.

4. **Separate ROTEA/SATEA in Exp1** shows confusion: 10% of ROTEA predicted as SATEA, and 3.3% of SATEA as ROTEA.

---

## 5. Confusion Matrix Analysis

### Exp1 (11-class) - Key Confusions on Test Set:

| True Class | Confused With | Rate |
|------------|---------------|------|
| ROTEA | SATEA | 10.0% |
| GBHEC | OTHERS | 33.3% |
| GBHEC | SATEA | 16.7% |
| OTHERS | SATEA | 15.8% |
| BRPEA | BRPEC | 9.8% |

### Exp2 (10-class) - Key Confusions on Test Set:

| True Class | Confused With | Rate |
|------------|---------------|------|
| TRHEA | LAGUA | 16.7% |
| GBHEC | LWBBA | 16.7% |
| LWBBA | GREGC | 14.3% |
| OTHERS | LAGUA | 7.1% |
| OTHERS | TRHEA | 7.1% |
| BRPEA | BRPEC | 7.3% |

### Interpretation:

- **Exp1** suffers from tern confusion (ROTEA↔SATEA), which is biologically expected as Royal and Sandwich Terns are visually similar.
- **Exp2** eliminates tern confusion but introduces new confusions involving TRHEA and rare white bird classes.
- **OTHERS class** is challenging in both experiments due to its heterogeneous composition.

---

## 6. Training Dynamics

### Convergence Speed:
- **Exp1:** Best val macro-F1 (0.806) at epoch 19
- **Exp2:** Best val macro-F1 (0.822) at epoch 17

### Final Epoch Comparison:

| Metric | Exp1 (ep 30) | Exp2 (ep 30) |
|--------|--------------|--------------|
| Train Acc | 99.2% | 99.7% |
| Train Loss | 0.027 | 0.013 |
| Val Acc | 88.6% | 93.2% |
| Val Macro-F1 | 0.801 | 0.775 |

**Note:** Both models show signs of overfitting (train acc ~99% vs val acc ~89-93%). Exp2 has lower training loss, indicating it finds the 10-class problem easier to optimize.

---

## 7. Improvement Over Orthomosaic-Split

Compared to the previous orthomosaic-level split (where rare classes were missing from val/test):

| Metric | Ortho-Split Exp1 | Crop-Split Exp1 | Improvement |
|--------|------------------|-----------------|-------------|
| Test Macro-F1 | 0.530 | **0.825** | +55.7% |
| Test mAP | 0.563 | **0.901** | +60.0% |
| Classes in Test | 7/11 | **11/11** | All present |

The crop-level split enables proper evaluation of all classes, dramatically improving macro-averaged metrics.

---

## 8. Recommendations

### When to use Exp1 (11-class):
- When distinguishing between Royal Terns (ROTEA) and Sandwich Terns (SATEA) is important
- For detailed species-level population counts
- When you need the best class-balanced performance (macro-F1)

### When to use Exp2 (10-class):
- When tern species can be grouped together for the application
- For higher overall prediction accuracy
- When computational simplicity is preferred (fewer classes)

### Future Improvements:
1. **Address GBHEC/OTHERS confusion:** Consider merging GBHEC into a broader "Heron Chicks" class or collecting more training data.
2. **TRHEA in Exp2:** The degradation suggests TRHEA benefits from the model learning tern-specific features that help distinguish it.
3. **Data augmentation:** Focus on rare classes (GBHEC, LWBBA, GREGC, GREGA) which have <100 samples.

---

## 9. Files Generated

### Exp1 (11-class):
- `runs_2025_exp1_11class/swin_cropsplit_mpc3000_ep30_lr0100_wd0100/`
  - `train.log` - Training logs
  - `curves.pdf` - Loss/accuracy curves
  - `val_test_cms.pdf` - Confusion matrices
  - `best.pt` - Best model checkpoint
  - `summary.json` - Final metrics

### Exp2 (10-class):
- `runs_2025_exp2_10class_terns/swin_cropsplit_mpc3000_ep30_lr0100_wd0100/`
  - Same structure as above

---

## 10. Conclusion

Both experiments achieve strong performance with the crop-level split ensuring proper evaluation across all classes. **Exp1 (11-class) is recommended** for applications requiring species-level granularity, achieving **82.5% test macro-F1** and **90.1% mAP**. **Exp2 (10-class)** is preferred when tern species can be grouped, achieving **93.0% overall accuracy** with fewer misclassifications.

The ~10% confusion between ROTEA and SATEA in Exp1 is biologically expected and may be acceptable depending on the application. If tern-level detail is not needed, Exp2 provides a cleaner, more accurate model.
