# Slicer HR-pQCT Toolbox

The Slicer extension has evolved from a single Timelapsed wrapper into an HR-pQCT toolbox. It keeps the core Python repositories as the source of truth while exposing interactive Slicer modules for running, reviewing, importing, exporting, and segmenting HR-pQCT data.

## Repository Direction

- Keep existing core Python repositories as the source of truth for algorithms, file formats, models, and command-line workflows.
- Use the Slicer extension repository for module UI, scene loading, workflow launchers, review tools, and lightweight Slicer-specific helpers.
- Keep the existing extension repository URL for continuity: `SlicerTimelapsedHRpQCT`.

Core algorithms remain in their Python packages. The Slicer toolbox should call those packages rather than copying registration, motion scoring, Scanco I/O, contour generation, or remodelling logic into Slicer modules.

## Current Modules

- `Timelapsed HR-pQCT`: run and review the longitudinal timelapsed workflow, including import, mask generation, timelapse registration, transform application, filling, remodelling analysis, interactive remodelling review, scenario export, and cohort row export.
- `Motion Scoring`: run prediction/review workflows from the MotionScore core package, configure local or downloaded model bundles, and export grading tables.
- `Scanco I/O`: import Scanco `.AIM` images as density/BMD, native Scanco values, mu, or HU; preserve editable AIM metadata on imported Slicer volumes; and export edited grayscale volumes or binary masks back to AIM.
- `Contours and Segmentation`: generate full, trabecular, cortical, and binary segmentation outputs from a selected HR-pQCT volume with radius, tibia, and knee presets plus standard Gaussian, Laplace-Hamming, and adaptive segmentation methods.

The modules appear in Slicer under the `HR-pQCT` category.

## Integration Contract

- Core packages own persistent dataset artifacts and registries.
- Slicer modules should write through public core APIs or CLIs.
- Shared Slicer utilities may understand core artifact paths, but should not create alternate sidecar state for the same concepts.
- The transform convention exposed to Slicer remains SimpleITK/LPS physical coordinates, moving -> fixed.

## Developer-Mode Loading

Use the helper script in the Slicer Python interactor:

```python
script = "<repo>/TimelapsedHRpQCTSlicer/scripts/link_local_toolbox_modules.py"
exec(open(script).read(), {"__name__": "__main__", "SCRIPT_PATH": script})
```

The helper registers all toolbox module paths:

- `TimelapsedHRpQCT`
- `MotionScoreHRpQCT`
- `ScancoIO`
- `HRpQCTSegmentation`

Restart Slicer after running the helper.
