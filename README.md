# TimelapsedRemodelling

Quantifying remodelling activity from time-lapsed HR-pQCT images of the distal radius or tibia  
Re-developed by Matthias Walle and collaborators, 2025  

---

## Overview

`TimelapsedRemodelling` is a Python 3.8 tool for automated analysis of longitudinal bone scans acquired with HR-pQCT.  
It performs:

- Preprocessing of scan and contour data
- Rigid registration of time points
- Segmentation-based remodelling detection
- Output of 3D masks, diagnostic figures, and per-region metrics

The tool is optimized for standardized file naming and organized input directories.

---

## Requirements

- Python 3.8 (due to `aim` support)
- macOS or Linux (tested on macOS 14.7, Apple M3, and Ubuntu 22.04)
- Conda or virtualenv recommended

---

## Installation

We recommend using a dedicated environment:

```bash
# Create and activate environment
conda create -n timelapse python=3.8 -y
conda activate timelapse

# Clone this repository
git clone https://github.com/wallematthias/TimelapsedRemodelling.git
cd TimelapsedRemodelling

# Install dependencies
pip install -e .
```

---

## File Naming Requirements (!)

Correct file naming is **critical** for this tool to associate scans and contours properly.

### Input scans must follow:
```
SUBJECT_SITE_TIMEPOINT.AIM
e.g., PT001_DR_T0.AIM, PT001_DR_T1.AIM .. 
```
DT LT RT TR TL = Tibia identifiers (these can be changed via `--tibia_identifiers` flag)

Otherwise = Treated as Radius (important for contouring) 

### Contours must use matching names with suffixes:
```
PT001_DR_T0_TRAB_MASK.AIM
PT001_DR_T0_CORT_MASK.AIM
```

If masks are not present, they will be generated using predefined logic and saved to the output folder as .mha files

---

## Usage

Run the processing pipeline for a single patient using:

```bash
remodell \
  PT001_DR_T0.AIM PT001_DR_T1.AIM \
  --output_path results \
  --result_pairs 0 1 \
```

### Required Arguments
- List of scan paths (`*.AIM`) in timepoint order (e.g., 0, 1, 2, ...)
- `--result_pairs`: Specify which timepoints to analyze as (baseline, followup) pairs (e.g., `0 1` or `0 1 1 2` for consecutive analysis `0 1 0 2` for baseline analysis)
- Note: The registration is always consecutive - the transforms are then compounded for the specific analysis. Registration is most precise between the closest timepoints. 

### Optional Arguments
- `--resolution`: Voxel spacing (e.g., `0.0607` mm)
- `--tibia_identifiers`: Strings used to identify tibia scans (default: `DT LT RT TR TL`)
- `--trabmask`, `--cortmask`: Filenames suffixes for contour loading

---

## Outputs

### Image Data (`.mha` files)
Each timepoint pair generates several `.mha` images:
- Naming follows `*_0_1_*.mha`, `*_1_2_*.mha` etc., indicating transformations from follow-up to baseline.
- Transformed image and mask types include:
  - `CORT_MASK`
  - `TRAB_MASK`
  
- Note you can check the .log files to see which image is associated with which index. 

- Remodelling labels are saved with the same convention (e.g., `remodelling_0_1.mha`).

> Note: `0_1` and `1_2` outputs are aligned to different baselines (0 and 1 respectively), so they are not directly overlaid. However, the **common region** analyzed is always defined relative to the baseline.

### Remodelling Metrics (`.csv`)
- One `.csv` file is generated **per patient**, named after the first timepoint (e.g., `PT001.csv`).
- Contains region-specific measurements:
  - `FVBV`: Formation volume fraction
  - `RVBV`: Resorption volume fraction
  - `BV`: Total bone volume at baseline

### Visualization for ParaView (`.vti`)
- A single `.vti` file is created for visual inspection in ParaView.
- Includes all registered images and masks aligned to the baseline frame of reference.
- Supports overlay-based exploration.

### Debug Overlays (`.png` files)
- Slices visualized and saved for quality control:
  - `*_remodelling.png`: Shows central slice of remodelling results
    - Orange = formation, purple = resorption
  - `*_masks.png`: Shows image slice with overlays of masks (trabecular, cortical, etc.)

### Transformations and Quality Control (`.tfm`, registration metrics)
- Each registration step outputs a `.tfm` file containing the rigid transformation matrix.
- The registration quality is recorded via the similarity metric (R value).
- If the R value is **low**, check visually whether:
  - Registration has failed
  - Apparent remodelling is due to misalignment
  - The scan includes **motion artifacts**
- For motion assessment, use the [MotionScoreCNN](https://github.com/wallematthias/MotionScoreCNN) tool.
  - Only include scans with **motion score ≤ 2** (especially important for tibia).

---

## Workflow

1. **Image and mask loading**  
   Automatically loads or generates contours using outer + inner segmentation.

2. **Registration**  
   Performs rigid registration across timepoints using masked regions.

3. **Remodelling analysis**  
   Classifies voxels as formation, resorption, or quiescence and outputs region-specific metrics.

---

## Mechanoregulation Analysis (coming soon)

We are currently extending the tool to support mechanoregulation analysis, which assesses how local mechanical strain correlates with bone remodelling. This feature will allow you to:

- Load finite element (FE) strain maps aligned to your timepoints
- Compute local strain percentiles per voxel
- Compare mechanical signal distributions across remodelling classes
- Output visualizations and .csv summaries of strain–remodelling relationships

Stay tuned for updates in future versions.

---

## Troubleshooting

- **Stalls or crashes at shutdown?**  
  Add this to the end of your script to avoid cleanup issues from native ITK libraries:
  ```python
  import os; os._exit(0)
  ```

- **Wrong masks associated?**  
  Double-check file naming — suffixes must match the `--trabmask` and `--cortmask` arguments.

---

## Citation

If you use this tool in a publication, please cite:

> Walle, M., Whittier, D.E., Schenk, D., Atkins, P.R., Blauth, M., Zysset, P., Lippuner, K., Müller, R. and Collins, C.J., 2023. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. Bone, 172, p.116780.

For related methodology, cite:

> Walle, M., Duseja, A., Whittier, D.E., Vilaca, T., Paggiosi, M., Eastell, R., Müller, R. and Collins, C.J., 2024. Bone remodeling and responsiveness to mechanical stimuli in individuals with type 1 diabetes mellitus. Journal of Bone and Mineral Research, 39(2), pp.85-94.

> Whittier, D.E., Walle, M., Schenk, D., Atkins, P.R., Collins, C.J., Zysset, P., Lippuner, K. and Müller, R., 2023. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. Bone, 176, p.116893.

> Walle, M., Gabel, L., Whittier, D.E., Liphardt, A.M., Hulme, P.A., Heer, M., Zwart, S.R., Smith, S.M., Sibonga, J.D. and Boyd, S.K., 2024. Tracking of spaceflight-induced bone remodeling reveals a limited time frame for recovery of resorption sites in humans. Science Advances, 10(51), p.eadq3632.

---

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.

---
