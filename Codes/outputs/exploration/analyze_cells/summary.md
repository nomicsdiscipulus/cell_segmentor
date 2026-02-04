# BCCD Cell Analysis Summary

**Analysis Date:** February 3, 2026
**Dataset:** BCCD (Blood Cell Count and Detection)
**Samples Analyzed:** 100 images from training set

---

## Overview

This analysis examines cell characteristics in the BCCD dataset using connected components analysis with size-based classification. The goal is to understand cell type distribution, size ranges, and data quality before implementing the segmentation model.

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total components detected | 9,036 |
| Valid cells (≥100 px) | 6,927 (76.7%) |
| Noise components (<100 px) | 2,109 (23.3%) |

---

## Cell Type Classification

Classification is based on component size in pixels:

| Cell Type | Count | Percentage | Size Range (px) | Description |
|-----------|-------|------------|-----------------|-------------|
| RBC | 6,397 | 92.3% | 1,000 - 12,000 | Red blood cells, pink discs |
| WBC | 527 | 7.6% | 12,000 - 20,000 | White blood cells, visible purple nuclei |
| Merged | 18 | 0.3% | > 20,000 | Overlapping/touching cells |
| Platelet | 0 | 0% | 100 - 1,000 | Not detected in dataset |
| Noise | 2,109 | - | < 100 | Artifacts, excluded from analysis |

---

## Size Statistics (Valid Cells Only)

### Cell Area (pixels)

| Statistic | Value |
|-----------|-------|
| Minimum | 1,127 |
| Maximum | 32,403 |
| Mean | 9,843 |
| Median | 7,720 |
| Std Dev | 2,769 |

### Estimated Cell Diameter (pixels)

Assuming circular cells: diameter = 2 × √(area / π)

| Statistic | Value |
|-----------|-------|
| Minimum | 40 px |
| Maximum | 202 px |
| Mean | 181 px |

---

## Classification Thresholds

The following pixel thresholds were used for size-based classification:

```
Noise:     < 100 px
Platelet:  100 - 1,000 px
RBC:       1,000 - 12,000 px
WBC:       12,000 - 20,000 px
Merged:    > 20,000 px
```

---

## Understanding the Raw Size Distribution

In the "Raw Size Distribution" plot (top-left of `bccd_size_distribution.png`), there is a solid rectangular block below 10³ (1000 pixels) on the log-scale x-axis. This represents the **noise components**:

| Size Range | Count | What it represents |
|------------|-------|-------------------|
| < 100 px | ~2,109 | Noise/artifacts (the solid block) |
| 100 - 1,000 px | 0 | No platelets detected |
| 1,000 - 12,000 px | ~6,397 | RBCs (the main peak) |
| > 12,000 px | ~545 | WBCs + merged cells |

**Why does it appear as a solid block?**

- There are 2,109 noise components with sizes < 100 pixels
- Most are extremely small (just a few pixels each)
- On a log-scale x-axis, these tiny values get compressed into the leftmost histogram bins
- Since ~2,109 components fall into these few bins, the counts reach ~2000, creating the solid rectangular appearance

**Sources of noise:**
- Small mask artifacts from image processing
- Edge fragments from incomplete cell boundaries
- Staining artifacts in the original blood smear images

The "Filtered Distribution" plot (top-right) shows the same data after removing noise (size ≥ 100 pixels), resulting in a cleaner normal distribution centered around 7,000-10,000 pixels.

---

## Key Findings

1. **RBC Dominance**: The dataset is predominantly composed of red blood cells (92.3% of valid cells), which is expected for blood smear images.

2. **No Platelets Detected**: Despite setting a platelet size range (100-1,000 px), no components fell within this range. Platelets are either:
   - Not present in the original images
   - Too small to be captured in the segmentation masks
   - Below the mask detection threshold

3. **Significant Noise**: About 23% of detected components are noise (artifacts <100 pixels). These should be filtered during preprocessing.

4. **Clear Size Separation**: RBCs and WBCs show distinct size ranges with minimal overlap, making size-based classification effective:
   - RBCs: centered around 7,720 px (median)
   - WBCs: typically >12,000 px with visible nuclei

5. **Merged Cells**: A small number of components (18) exceed 20,000 px, indicating overlapping or touching cells that were not separated in the masks.

---

## Visualizations Generated

| File | Description |
|------|-------------|
| `bccd_cells_by_type.png` | Single image with cells colored by estimated type |
| `bccd_size_distribution.png` | Dataset-wide histograms and statistics |
| `bccd_sample_cells.png` | Cropped individual cells sorted by size |

---

## Implications for Segmentation Model

1. **Binary to Instance**: The BCCD masks are binary (foreground/background). Connected components analysis is required to separate individual cell instances.

2. **Noise Filtering**: Apply minimum size threshold (~100 px) to remove artifacts.

3. **No Cell Type Labels**: The dataset does not provide cell type annotations. Size-based heuristics or a separate classification head may be needed if cell type prediction is required.

4. **High Cell Density**: Images contain 13-295 cells (mean ~90), requiring the model to handle crowded scenes with potential occlusions.

5. **Image Resizing**: Original images are 1600×1200. Significant resizing will be needed for model input (e.g., 224×224 or 518×518), which will affect cell sizes proportionally.
