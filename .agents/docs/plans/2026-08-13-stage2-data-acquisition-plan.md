# Stage 2 Data Acquisition Implementation Plan

## Goal

Execute the Stage 2 (platanus / other-tree) data acquisition per
`.agents/docs/specs/2026-08-13-stage2-data-source-investigation.md`, with the
deployment GSD baseline fixed at 100 m AGL (~9.4 cm/px).

## Fixed Baseline

| Item | Value |
|---|---|
| Camera | DJI Matrice 4T wide stream, 82 deg DFOV, 1440x1080 NV12 @ 30 fps |
| Model input | 1280x1280, letterbox content 1280x960 (scale 0.8889) |
| Flight altitude | 100 m AGL (below the 120 m ceiling with margin) |
| Nominal GSD | ~9.4 cm/px (calibrate on first flight) |
| Crown size in input | 8-15 m platanus crown = 76-142 px |
| Data acceptance band | 9-15 cm direct; 25 cm supplementary only |

## Data Source Tiers

### Tier 1: request-based, high-fidelity geometry

| Source | GSD | Content | Priority |
|---|---|---|---|
| Sydney ArborCam study | 12 cm | 439 Platanus crowns, pixel masks, airborne imagery | 1st |
| NEON DTA + DP3.30010.001 RGB | 10 cm | 103 PLOC crowns (9 human boxes, 67 detector boxes) | 2nd, startable now |
| Enschede allergenic tree mapping | 25 cm | crown polygons + municipal inventory, YOLOv11-seg prep | 3rd, supplementary |

### Tier 2: request-free scale providers

| Source | Role |
|---|---|
| NYC 2015 Street Tree Census + Esri tiles | volume labels: 87,014 platanus points; auto-delineate crowns with the Stage 1a/1b model; exhaustive two-class labeling inside each crop window from inventory points |
| BAMFORESTS (HF mirror `CanopyRS/BAMForests`) | optional Stage 1b generic pretraining, CC BY 4.0 |

### Excluded

TreeCrown-MM (no Platanus in the 285-species table), WHU-STree
(street-level), GatorSense 224 px crops as mask supervision, NEON fallback
squares, Pasadena REGISTREE (points only + Google ToS risk).

## Application List

1. **NEON API token**: register at neonscience.org; no approval delay.
   Then download the fixed-release RGB tiles for the PLOC crowns after
   fixing the DELA CRS error (zone 16, not 15).
2. **Sydney ArborCam email**: request imagery + Platanus masks from the
   paper authors (Carnegie et al., Urban For. Urban Green. 81, 2023);
   state graduation-project research use and cite the paper.
3. **Enschede email**: request crown polygon shapefiles + municipal
   inventory from the GitHub repo authors / Municipality of Enschede.
4. **No application needed**: NYC census (Socrata API), BAMFORESTS HF,
   Esri World Imagery tiles (prefer Esri over Google to avoid the Google
   Maps ML-training prohibition).

## Per-Source Build Steps

### NYC + Esri (volume)

1. Query the Socrata API for `spc_common='London planetree'` plus the top
   negative species (honeylocust, callery pear, pin oak, Norway maple, ...).
2. Fetch Esri tiles (~15 cm zoom) around each point; crop >= 256 px scenes.
3. Delineate crowns with the Stage 1a model (or SAM) inside each crop.
4. Label every inventory point in the crop as platanus / other-tree so
   windows are exhaustive with respect to the inventory.
5. Manually verify a 5-10% subset (crown match, removals, occlusions).
6. Record source manifest: census revision, tile provider, crop hashes.

### NEON (precision pilot)

1. Pin the DTA gpkg revision and SHA-256 (`eeffb0c0...43c61f`).
2. Extract PLOC rows; rebuild DELA geometry from original VST coordinates
   (UTM zone 16) before any download.
3. Download the 10 cm RGB tiles (DP3.30010.001, 1 km tiles) with the NEON
   token for the 9 human-box crowns (SCBI first: 5 human boxes).
4. Refine crown polygons manually; keep the 67 detector boxes as a
   candidate pool, drop the 27 fallback squares as mask supervision.
5. Split by `individual` / `siteID + plotID`; use a site holdout.

### Sydney (if granted)

1. Convert the provided masks/imagery to YOLO-seg format.
2. Re-split per the paper's spatially independent validation design.
3. Use directly as a 12 cm anchor set (closest match to 9.4 cm).

### Enschede (if granted)

1. Convert crown polygons + inventory to two-class YOLO-seg format.
2. Treat as 25 cm supplementary data: upsample/augment to the 9-15 cm
   band; never the sole fine-tuning source.

## GSD Alignment

- Anchor: 10 cm. All training tiles are resampled into a 9-15 cm band
  before fine-tuning; ultralytics scale/mosaic augmentation covers the
  residual spread.
- The Stage 1a checkpoint (OAM-TCD, ~2 cm) is an initializer only; its
  scale gap is handled by fine-tuning, not by inference.

## Training Flow

```
Stage 1a best.pt (existing)
  -> [optional] Stage 1b BAMFORESTS generic pretraining
  -> external two-class dataset (NEON pilot + NYC volume [+ Sydney])
  -> GSD alignment to 9-15 cm
  -> two-class fine-tune (single_cls: false, 1280x1280)
  -> local Xi'an M4T @ 100 m capture (small set) -> final fine-tune
  -> spatially independent local evaluation + release gates
```

## Milestones

1. NEON token obtained; PLOC pilot tiles downloaded and DELA fixed.
2. Application emails sent (Sydney, Enschede).
3. NYC volume set built with 5-10% verification; manifest recorded.
4. Two-class fine-tuning dataset assembled and split (individual/site
   group-level disjointness).
5. Local capture campaign defined (100 m, wide stream, nadir).

## Risks

- Sydney / Enschede requests may be refused; NYC + NEON alone still
  provide thousands of weak/semi labels, sufficient for a graduation
  baseline but not for release-grade platanus fidelity.
- NYC census (2015) vs current tile imagery has a time offset; removals
  and regrowth require verification.
- Liveview GSD depends on the unverified 82-degree FOV assumption;
  calibrate with the laser rangefinder on the first flight.
- The onboard letterbox (1280x960 content) must match training
  letterbox semantics; verify with the training-side validation set.

## References

- `.agents/docs/specs/2026-08-13-stage2-data-source-investigation.md`
- Onboard repo: `manifold-3-vision-detect` `src/capture/liveview_capture.cpp`,
  `src/inference/preprocess.cpp`, `docs/project-status.md`
- NYC census API: https://data.cityofnewyork.us/resource/uvpi-gqnh.json
- NEON RGB: https://data.neonscience.org/data-products/DP3.30010.001
- NEON DTA: https://huggingface.co/datasets/weecology/neon-tree-crowns-dta
- Carnegie et al. 2023, Urban For. Urban Green. 81:127859
- Enschede: https://github.com/klavdix12/urban-allergenic-tree-mapping
