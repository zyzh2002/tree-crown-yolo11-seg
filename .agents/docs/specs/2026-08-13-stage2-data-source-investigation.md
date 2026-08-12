# Stage 2 Data Source Investigation

## Status

Research record for Stage 2 (platanus / other-tree) data acquisition. The
candidate datasets were investigated in detail on 2026-08-13, including full
paper reads, Hugging Face metadata audits, and direct downloads of dataset
artifacts. No Stage 2 training has started; the acquisition decisions below
are recommendations pending data requests and user confirmation.

This document extends `2026-08-07-platanus-other-tree-training-strategy.md`,
which remains the authority for the training flow, the two-class ABI, and the
release gates.

## Problem Statement

Stage 2 requires top-down RGB imagery with individual crown geometry and
genus-level `platanus` species labels. Local Xi'an imagery is not yet
collectable. The repository must therefore source either (a) a packaged
dataset that already pairs aerial imagery, crown geometry, and Platanus
species labels, or (b) an authoritative municipal inventory plus a separate
aerial imagery source that can be joined by coordinates.

## Evaluation Criteria

A candidate is usable for Stage 2 only if it satisfies all of:

1. **Top-down aerial RGB** (orthomosaic, airborne, or satellite at
   <= 25 cm GSD preferred; street-level panoramic data is rejected).
2. **Individual crown geometry** (polygon, mask, or at minimum a bounding
   box); inventory points alone do not supervise a mask head.
3. **Genus-level Platanus labels** with traceable evidence (authoritative
   municipal inventory, expert ground survey, or expert labeling).
4. **Accessible license and distribution** for a graduation project,
   including non-public datasets obtainable by author request.

## Investigated Datasets

| Dataset | Aerial RGB | Crown geometry | Platanus | Verdict |
|---|---|---|---|---|
| BAMFORESTS (DLR / HF mirror) | yes, 1.6-1.8 cm | polygons | no | Generic pretraining only (Stage 1b) |
| TreeCrown-MM (HF `huafei-77`) | yes | masks | no | Excluded |
| NEON DTA (`weecology/neon-tree-crowns-dta`) | via NEON DP3.30010.001, 10 cm | boxes/fallbacks | yes, 103 PLOC | Weak labels + manual refinement pilot |
| GatorSense NeonTreeClassification | 224 px crops only | none | yes, PLOC | Classification aid only; shares DTA lineage |
| REGISTREE Pasadena Urban Trees | 15 cm Google tiles | points only | unconfirmed | Downgraded; Google ToS risk |
| NYC 2015 Street Tree Census | no imagery | none | 87,014 points | Label source only; imagery must be fetched |
| opentrees.org | no imagery | none | varies by city | Label aggregator only |
| Enschede allergenic tree mapping | 25 cm RGB/CIR | polygons | yes | Promising; labels need request |
| Sydney ArborCam study | 12 cm airborne | masks | 439 trees | Data not public; author request |
| WHU-STree | street-level panoramic | 3D instances | yes | Excluded: not aerial |

## Key Findings per Source

### BAMFORESTS

Full paper read (Troles et al., Remote Sens. 16, 1935, 2024; PDF via DLR
elib). 27,160 crowns, 105 ha, GSD 1.6-1.8 cm, COCO format, fixed benchmark
split by hectare plots. The paper states the species and vitality labels were
used only to balance splits: "we only publish the ITC shapes and not all
existing labels"; BAMFORESTS-2 will publish them. The HF mirror
`CanopyRS/BAMForests` was audited with polars across all 39 train parquet
shards (1,438 rows): `category` contains only `"tree"`. Species mix
(Pinus/Fagus/Quercus/Picea/Larix/Pseudotsuga/Abies/Other) contains no
Platanus. Stage 1b role unchanged: generic single-class pretraining.

### TreeCrown-MM

330K crowns, 285 species, five ecosystems including Urban-Mixed, RGB + CHM
+ masks + captions. The supplement species table was downloaded and searched:
zero occurrences of Platanus/sycamore. Excluded.

### NEON family

`weecology/neon-tree-crowns-dta` (HF) was downloaded (`neon_crowns_dta.gpkg`,
27.5 MB, SHA-256 `eeffb0c0...43c61f`) and audited directly:

- 41,738 rows, 234 taxonIDs, 38 sites.
- PLOC (`Platanus occidentalis L.`) = 103 rows across 9 sites
  (BLAN 37, SERC 31, DELA 12, SCBI 8, LENO 6, GRSM 5, TALL 2, KONZ 1,
  UKFS 1), years 2019-2022.
- Geometry semantics: 67 DeepForest detection boxes (median area 49.7 m²,
  median score 0.45, 18 below 0.3), 27 fallback 2 m x 2 m squares, 9 human
  bounding boxes, **0 fine crown polygons**.
- Data lineage risk: DeepForest boxes were matched to NEON Vegetation
  Structure stems by a nearest-center rule; field stem coordinates carry
  geolocation uncertainty and can match a neighboring crown.
- DELA site bug: DELA is stored with `crs_epsg=32615` (UTM zone 15) but the
  site is in zone 16; all 12 DELA PLOC rows need geometry reconstruction
  before imagery download.
- Imagery: NEON `DP3.30010.001` RGB GeoTIFFs, 1 km tiles, 10 cm GSD for
  year >= 2017 (25 cm before). Downloads now require a NEON API token
  (unauthenticated requests return 403). New downloads are CC BY 4.0.
- Leakage controls: split by `individual` / `siteID + plotID`; no cross-year
  duplicates of the same tree; a site holdout is recommended.

Conclusion: usable as a weak-label bootstrap and manual-annotation pilot
(e.g., SCBI has 8 PLOC including 5 human boxes), not as mask ground truth.
The associated `GatorSense/NeonTreeClassification` package is 224 px
classification crops sharing the same DTA individuals (133 PLOC rows, 47
unique individuals, duplicate RGB within year); it is a classification aid
only and must be deduplicated against DTA. The HF repo
`ritesh313/NeonTreeClassificationData` is empty (README only).

### REGISTREE / Pasadena Urban Trees

Official page: ~30,000 trees labeled by geo-location and species with dense
aerial + street-view imagery; request by email (research use only). CVPR
2016 paper: 28,678 aerial tiles at ~15 cm, aerial date 2015-03, inventory
~80,000 trees, species-recognition subset 5,205 trees / 18 species. Two
blockers: (1) the paper says the municipal labels "are points rather than
bounding boxes" - no crown geometry is shipped; (2) current Google Maps
Platform terms prohibit bulk download/storage and ML training on Google Maps
content, so the inherited 2014-2015 tiles are legally risky unless the
dataset maintainers confirm otherwise. Keep as a conditional candidate only
after an actual data package review.

### NYC 2015 Street Tree Census

Queried live via the Socrata API: 683,788 records; `London planetree` =
87,014 (top species). Fields include species, lat/lon, DBH, health. Public
domain. No imagery is included. This is the strongest label source; it must
be joined with an aerial tile source (Esri/Google/Bing) and verified, e.g.,
the Camden workflow (Waters et al., ICML 2021: Google Maps static crops +
Camden inventory, six species including London Plane; best model ~60-69%
accuracy on 200 px crops - the crop size and label noise were limiting).
The Sydney ArborCam study (Carnegie et al., Urban For. Urban Green. 81,
2023) demonstrates the ceiling of the inventory + airborne imagery route:
439 ground-mapped Platanus crowns, 12 cm RGB, instance-segmentation
accuracy 95.2% for Platanus, but the data are not public.

### opentrees.org

13.9M aggregated inventory trees, 192 sources, 19 countries; vector tiles
of points with species. Contains no imagery of any kind; useful only as an
alternative label aggregator.

### Enschede allergenic tree mapping

GitHub `klavdix12/urban-allergenic-tree-mapping` (code DOI
10.5281/zenodo.20600808). Explicitly targets five genera including
Platanus, with 2022 25 cm RGB + CIR aerial imagery (public, Dutch
GeoTiles), AHN4 LiDAR, municipal inventory, and ground-truth crown polygon
shapefiles, plus a YOLOv11s-seg data preparation pipeline. The crown
polygons and inventory are held by the Municipality of Enschede,
4TU.HERITAGE, and the University of Twente and are not in the repo. This is
the closest packaged match to the Stage 2 requirement; the next step is a
data request to the authors / municipality.

### Sydney ArborCam

Paper evidence only (see above); airborne data and masks belong to
ArborCarbon / partners. Author request is the only path.

### WHU-STree

Excluded per the training strategy spec: street-level MMS point clouds and
panoramic images, not aerial. Confirmed the species list includes
`Platanus x acerifolia` (PA), so it may serve as a species reference only.

## Decision

1. **Primary route: NYC census labels + aerial tile imagery.** 87k Platanus
   points are joined with Esri/Google aerial tiles (>= 256 px crops around
   each point, larger than the Camden 200 px crops), crowns are delineated
   by the Stage 1a/1b model or SAM, and a 5-10% subset is manually verified
   against Street View/overhead imagery. Label evidence: authoritative
   municipal inventory, satisfying the strategy spec's traceability gate.
2. **Small-pilot route: NEON PLOC.** Fetch the 10 cm tiles for the 9
   human-box crowns (and optionally the 67 DeepForest boxes after manual
   review), fix the DELA CRS error first, and produce a small high-quality
   platanus mask set for scale validation.
3. **Requests to file:** Enschede crown polygons (primary), Sydney ArborCam
   data (secondary), REGISTREE Pasadena package (conditional, license
   check).
4. **Excluded:** TreeCrown-MM (no Platanus), WHU-STree (viewpoint),
   GatorSense crops as mask supervision (classification only), fallback
   squares from DTA as mask supervision.

## Open Questions

- Deployment GSD: determines whether 15-25 cm sources can be used
  directly or need mixed-GSD augmentation with local 2-5 cm imagery.
- Whether the NYC census-to-tile join has acceptable positional noise at
  the chosen tile source (tree removals, GPS error, occlusions).
- Enschede data holders' willingness to share crown polygons for a
  graduation project.

## References

- Troles et al., "BAMFORESTS", Remote Sens. 16(11):1935, 2024.
  https://doi.org/10.3390/rs16111935
- Waters et al., "Urban Tree Species Classification Using Aerial Imagery",
  ICML 2021 workshop. https://arxiv.org/abs/2107.03182
- Carnegie et al., "Airborne multispectral imagery and deep learning for
  biosecurity surveillance", Urban For. Urban Green. 81:127859, 2023.
  https://doi.org/10.1016/j.ufug.2023.127859
- Branson et al., "From Google Maps to a Fine-Grained Catalog of Street
  trees", ISPRS J. Photogramm. Remote Sens. 135:13-30, 2018.
  https://doi.org/10.1016/j.isprsjprs.2017.11.008
- Wegner et al., "Cataloging Public Objects Using Aerial and Street-Level
  Images - Urban Trees", CVPR 2016.
- Weinstein et al., "Individual canopy tree species maps for the National
  Ecological Observatory Network", PLOS Biology 22(7):e3002700, 2024.
  https://doi.org/10.1371/journal.pbio.3002700
- NEON Tree Crowns DTA: https://huggingface.co/datasets/weecology/neon-tree-crowns-dta
- NEON RGB DP3.30010.001: https://data.neonscience.org/data-products/DP3.30010.001
- NYC 2015 Street Tree Census:
  https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh
- Pasadena Urban Trees: https://registree.ethz.ch/publications-and-dataset.html
- opentrees.org: https://opentrees.org/
- Enschede: https://github.com/klavdix12/urban-allergenic-tree-mapping
- WHU-STree: https://github.com/WHU-USI3DV/WHU-STree
