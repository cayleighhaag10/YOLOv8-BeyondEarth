# BoulderNet / CS231N Research Summary
*Last updated: June 4, 2026*

---

## Project Overview

Replace YOLOv8's mask head with SAM2 for precise boulder segmentation in HiRISE planetary imagery. Pipeline: YOLO detects bounding boxes → SAM2 produces pixel-accurate segmentation masks. Includes SAM2 fine-tuning on boulder data.

**Raster:** `M1221383405.tif` — Moon, LRO NAC, 0.634 m/px, 12816×55680px  
**GT dataset:** Prieur et al. 2023 (Zenodo 8171052) — 5 labeled test tiles  
**Repo:** `/Users/cayleigh/Desktop/BoulderNet/YOLOv8-BeyondEarth`  
**Paper notebooks:** `resources/nb/paper/`

---

## Core Results

### AP Metrics (Table 1) — per-tile eval, overlap=0, mask IoU

| Model | AP50 | AP50:95 |
|---|---|---|
| YOLOv8 | 0.500 | — |
| SAM2 zero-shot | 0.514 | — |
| SAM2 fine-tuned | **0.525** | — |
| SAM2-auto | 0.140 | — |

SAM2 fine-tuned beats YOLO. SAM2-auto is poor — it runs without YOLO box prompts and misses many boulders.

**Caveat:** Full-raster eval reverses this ranking (YOLO wins at AP50=0.624 vs SAM2 0.541) because YOLO used a looser NMS threshold (0.2 vs 0.5) and larger masks inflate IoU at the 0.5 threshold. The per-tile eval is the controlled, fair comparison.

### Detection Counts (full raster, post-NMS)
| Model | Total detections | Elongated (AR 1.2–2.0) | Pass rate |
|---|---|---|---|
| YOLOv8 | 51,015 | 26,165 | 51% |
| SAM2 zero-shot | 53,924 | 10,219 | 19% |
| SAM2 fine-tuned | 55,204 | 10,496 | 19% |
| SAM2-auto | 31,122 | 17,533 | 56% |

SAM2 produces more total detections but fewer elongated ones — because SAM2 masks are tighter and more circular. Not an NMS artifact.

---

## Orientation Artifact — Full Investigation

### Root Cause (confirmed)
YOLOv8 and SAM2 masks are **pixel-aligned binary polygons** (staircase contours). When `EllipseModel` fits an ellipse to these H/V-dominated boundaries, it returns angles snapping to 0°/45°/90°/135° regardless of true boulder orientation. This is a rasterization artifact, not a model defect.

**Evidence:**
- Synthetic rasterization test: smooth ellipses at random angles → rasterize → staircase polygon → orientation snaps to grid-aligned angles
- Size stratification: artifact weakens but never disappears even for large boulders (>20px)
- All models (YOLO, SAM2, SAM2-ft, SAM2-auto) show the artifact

### Pipeline (current)
```
mask polygon (staircase)
  → segmentize(poly, res)          # adds points along edges
  → fitEllipse(EllipseModel)       # algebraic least-squares fit
  → minimum_rotated_rectangle()    # on smooth 128-pt ellipse polygon
  → boulder_row() → angle180       # azimuth from North, 0–180°
```

### Convention
- `theta` from EllipseModel: math angle CCW from East, [−90°, 90°]
- `angle180` from boulder_row: geographic azimuth CW from North, [0°, 180°]
- Relationship: `angle180 ≈ 90° − theta` (mod 180)

### Minimum Area Filter
`AREAL_THRESHOLD = (res² × 4.74²)` ≈ 9 m² at 0.634 m/px  
4.74 pixels is the labeling convention from Prieur et al. (dataset is called "mask-5px").  
**Important:** apply to GT only, NOT to predictions — SAM2 tight masks get cut unfairly.

---

## Methods Tried — Orientation Estimation

### Diagnostic methods (paper_01b)

| Method | D-stat (medium) | Notes |
|---|---|---|
| 1. segmentize → ellipse → MRR (baseline) | 0.033 | Reference |
| 2. Ellipse θ direct (skip MRR) | 0.033 | Identical to baseline — MRR is NOT the issue |
| 3. No segmentize → ellipse → MRR | 0.054 | Nearly identical — segmentize is NOT the main cause |
| 4. Polygon moments (Green's theorem) | 0.119 | **WORSE** than baseline — Green's theorem uses boundary edges, same H/V bias |

**Key finding:** Methods 1–3 are essentially identical. The artifact is baked into the staircase polygon before any of these steps. Segmentize neither helps nor hurts meaningfully.

### Correction attempts on per-tile data (paper_03)

| Method | KS D | p-value | 90° spike ratio | Verdict |
|---|---|---|---|---|
| GT (Prieur et al.) | 0.063 | 0.416 | 2.25× | Reference |
| Baseline (binary mask) | 0.071 | 0.115 | 1.55× | Reference |
| Gaussian blur (σ=2) | 0.077 | 0.157 | 3.04× | **FAIL** — staircase reintroduced at re-threshold |
| Shapely smoothing (buffer trick) | 0.071 | 0.098 | 1.61× | FAIL — polygon vertices still on pixel grid |
| Exp 6: SAM2 logit isocontour | 0.090 | 0.034 | 0.58× | FAIL — logit map is 256×256 grid, same issue |
| Exp 8: soft-mask image moments | 0.083 | 0.039 | 1.67× | FAIL |
| Exp 9: point-prompt reprompting | 0.113 | 0.006 | 1.28× | **WORSE** than baseline |
| **Exp 7: Canny gradient** | **0.051** | **0.566** | 2.29× | **BEST** — closest to GT, p indistinguishable from uniform |

**Caveat:** n~250 from 5 test tiles is too small to show clear spikes or clear fixes. All differences are marginal. The story is cleaner at scale (n=26k from full raster) but we only have per-tile data for the corrections.

### Exp 7 (Canny gradient) — the approach that works
Run Canny on the **full unmasked tile image** (NOT the masked crop), restrict detected edge pixels to the dilated mask region, fit ellipse to edge pixels using `cv2.fitEllipse`.

Key insight: never zero out the exterior before computing gradients — that creates artificial H/V edges at the staircase boundary. Compute on the full tile, sum only over interior.

Implementation: `orient_exp7` in `resources/nb/paper/fig3_fig4_corrections_gradient.ipynb`

### Structure tensor (paper_01c) — needs re-running with fix
**Bug found:** normalizing image and zeroing exterior pixels (`img_normed[~interior] = 0.0`) before Sobel creates artificial H/V gradients at the staircase boundary that dominate the tensor. **Fix applied** (June 4): compute Sobel on full tile without zeroing, restrict sum to interior pixels. Not yet re-run after fix.

### NMS sweep (paper_00b)
- Higher NMS IoU threshold = LESS aggressive = MORE detections kept
- NMS=0.1 → 52,162; NMS=0.4 → 53,184 for SAM2 zero-shot (~2% variation)
- NMS threshold is NOT the cause of low SAM2 elongated count
- SAM2 just produces rounder masks → fewer pass AR 1.2–2.0 filter

---

## Synthetic Ellipse Experiment — Orientation Debugging (June 2026)

**Notebook:** `resources/nb/paper/synthetic_ellipse_orientation.ipynb`  
**Setup:** N=300 synthetic images, one ellipse each (a=15, b=10, AR=1.5) at random orientations in [0°, 180°). SAM2 prompted with GT bounding boxes. Orientation measured via paper pipeline (fitEllipse → MRR → boulder_row, angle in geospatial convention).

### Main finding: the orientation artifact is a binary rasterization property, not a general consequence of noise or irregularity

All SAM2 variants recover orientation with MAE < 2° on clean synthetic ellipses, and orientation histograms are flat (KS D ≈ 0.04, p > 0.6) — **no cardinal spikes**. This confirms the paper's claim: the 0°/45°/90° spikes seen in real imagery are not a property of the measurement pipeline math, but specifically of pixel-aligned binary mask contours (YOLO's staircase mask head).

### Image noise sweep (SAM2 zero-shot, GT bbox)

| σ (pixel) | n valid | MAE (°) | KS D | p |
|---|---|---|---|---|
| 0 | 150/150 | 0.3 | 0.047 | 0.882 |
| 5 | 150/150 | 0.3 | 0.045 | 0.906 |
| 15 | 150/150 | 0.3 | 0.047 | 0.882 |
| 30 | 150/150 | 0.3 | 0.047 | 0.877 |
| 60 | 150/150 | 0.3 | 0.046 | 0.897 |
| 100 | 148/150 | 33.7 | 0.046 | 0.905 |

SAM2 is completely robust to image noise up to σ=60 (SNR ≈ 2.5 for our synthetic ellipses). At σ=100, errors grow to 33.7° but the histogram stays flat — errors are random, not cardinal-biased. Image degradation does not introduce orientation bias.

### Boundary noise sweep (radial perturbation, SAM2 zero-shot, GT bbox)

Boundary noise is applied radially (each point scaled outward/inward from center) to prevent self-intersections, simulating irregular boulder shapes.

| σ (world units) | n valid | MAE (°) | KS D | p |
|---|---|---|---|---|
| 0.0 | 150/150 | 0.3 | 0.045 | 0.907 |
| 0.3 | 150/150 | 0.5 | 0.042 | 0.949 |
| 0.7 | 127/150 | 45.2 | 0.093 | 0.212 |
| 1.5 | 22/150 | 46.8 | 0.176 | 0.451 |
| 3.0 | 0/150 | — | — | — |
| 6.0 | 0/150 | — | — | — |

Two regimes: below σ≈0.5 world units (≈3 pixels) the pipeline is stable; above it, the AR filter [1.2, 2.0] begins rejecting distorted shapes and surviving measurements are noisy (~45° MAE). **Critically, even at σ=0.7–1.5 where things are clearly breaking, the histogram remains non-significantly non-uniform (p=0.21, p=0.45) — no cardinal spikes.** Boundary irregularity introduces random orientation errors, not systematic bias.

### Conclusion (use in paper / future work)

> *"Boundary irregularity and image noise degrade orientation accuracy but do not introduce cardinal bias. Systematic spikes at 0°/45°/90°/135° are unique to pixel-aligned binary mask representations and are not a general property of noisy or irregular shapes."*

This supports the paper's framing that the Canny gradient fix (Exp 7) is the right direction: bypassing the binary contour entirely eliminates the artifact at its source.

### Other notes from this experiment
- YOLO detects 0/300 synthetic images (domain gap — trained on real planetary imagery only). For YOLO mask quality tests, use real imagery.
- SAM2 fine-tuned is less stable on synthetic data than zero-shot (134/300 vs 300/300 pass AR filter, median AR=1.52 but high variance). Domain mismatch: fine-tuning on narrow planetary boulder distribution reduces robustness to out-of-distribution inputs. Zero-shot SAM2 (trained on 1B diverse images) generalizes better.
- Coordinate convention: `make_smooth_ellipse` theta is math convention (CCW from East); `boulder_row` angle180 is geospatial (CW from North). Relationship: `angle180 = (90 - theta_math) % 180`.

---

## What We Concluded NOT to Include in the Paper

- fig3 (corrections fail): n too small, histogram noise overwhelms signal
- fig4 (Canny gradient works): marginal improvement at n~250, not convincing visually
- The "fix" narrative: evidence is weak at available sample sizes
- Scatter plot from fig2 synthetic experiment: interpretation is unclear

---

## Paper Structure (current plan)

1. **Table 1** — AP metrics (per-tile, clean eval): SAM2-ft > SAM2 > YOLO
2. **Fig 1** — Orientation histograms (full raster): artifact persists across all models
3. **Fig 2** — Synthetic experiment (two histograms only, drop scatter): explains mechanism
4. **Qualitative figures** — already have these

**Narrative:** SAM2 improves segmentation accuracy (AP). Orientation artifact is a fundamental property of pixel-aligned binary mask pipelines, persists across all models, and is a known limitation pointing to future work.

---

## Open Questions / Known Issues

1. **Full-raster SAM2 count discrepancy**: older memory noted ~14K SAM2 vs ~44K YOLO on full raster. Current session shows 53K SAM2 vs 51K YOLO. May have been from a different/earlier run. Worth verifying which is correct.

2. **SAM2-auto AP is very low (0.140)**: expected — it runs without YOLO box prompts so it misses many boulders and generates many false positives. Not a fair comparison to the prompted models.

3. **Score columns in shapefiles**: YOLO uses `confidence`, SAM2 uses `score` (predicted IoU). SAM2-auto may have different or no score column. All scores equal to 1.0 would prevent AP from sweeping the precision-recall curve properly.

4. **Structure tensor after zero-fill fix**: paper_01c was fixed June 4 but not re-run. Results unknown.

---

## Future Improvements

### Short-term (feasible now)
- Re-run structure tensor in paper_01c with zero-fill fix and compare to baseline
- Run Canny gradient (Exp 7) on full-raster data (26k boulders) to get statistically powered result
- Verify score columns exist and are meaningful for all model variants
- Per-tile AP broken down by tile ID to check for outlier tiles driving results

### Medium-term (research directions)
- **Larger fine-tuning dataset**: SAM2 was fine-tuned on a small set. More labeled data should improve both AP and mask quality
- **Orientation-aware evaluation**: current AP uses mask IoU at fixed thresholds. Could add orientation accuracy as a secondary metric
- **Canny gradient at scale**: run on full raster to properly evaluate whether it reduces the orientation bias at n=26k
- **SAM2 with better prompts**: instead of YOLO bounding boxes, use YOLO mask centroids as point prompts — may give tighter, better-calibrated masks

### Long-term (from advisor, May 2026)
- **Super-resolution model** (PSR-BoulderNet, ScienceDirect): allows detecting smaller boulders AND produces less blocky outlines → better orientations
- **Mask R-CNN comparison**: if Mask R-CNN gives different orientation distributions, confirms the artifact is YOLO-specific rather than general segmentation artifact
- **Continuous mask representation**: avoid binarization entirely — work with soft logit maps for orientation estimation (Exp 8 explored this but at 256×256 grid resolution which is too coarse)

---

## Key File Locations

| What | Where |
|---|---|
| Paper notebooks | `resources/nb/paper/` |
| Diagnostic notebooks | `resources/nb/paper_01b_*.ipynb`, `paper_01c_*.ipynb` |
| SAM2 predict pipeline | `src/YOLOv8BeyondEarth/SAM2_predict.py` |
| Orientation pipeline | `shptools/src/shptools_BOULDERING/geometry.py` (fitEllipse), `geomorph.py` (boulder_row) |
| Full-raster predictions | `~/tmp/YOLOv8BeyondEarth/exp_yolo_256/`, `exp_sam2_256/`, `exp_sam2_finetuned_256/` |
| Per-tile predictions | `~/tmp/YOLOv8BeyondEarth/exp_tile_eval/{model}_{tile_id}/` |
| GT dataset | `/scratch/users/cayleigh/Apr2023-Mars-Moon-Earth-mask-5px/preprocessing/test/` |
| Fine-tuned SAM2 weights | `/scratch/users/cayleigh/sam2_finetuned/sam2_boulder_best.pt` (best ep17, val=0.1056) |
| PowerPoint | `/Users/cayleigh/Desktop/BoulderNet_Orientation_Bug_Presentation.pptx` |
