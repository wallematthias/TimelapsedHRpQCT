# Slicer HR-pQCT Toolbox Roadmap

This roadmap proposes a new umbrella Slicer extension repository for HR-pQCT tools. The current TimelapsedHRpQCT and MotionScoreHRpQCT Slicer extensions should remain stable while the toolbox is designed and introduced.

## Repository Direction

- Create a new umbrella Slicer extension repository, for example `SlicerHRpQCTToolbox`.
- Keep existing core Python repositories as the source of truth for algorithms, file formats, models, and command-line workflows.
- Use the toolbox repository for Slicer module UI, scene loading, workflow launchers, review tools, and shared Slicer helpers.

Core algorithms remain in their Python packages. The Slicer toolbox should call those packages rather than copying registration, motion scoring, Scanco I/O, contour generation, or remodelling logic into Slicer modules.

## Proposed Modules

- TimelapsedHRpQCT module: run and review the longitudinal timelapsed workflow, including import, masks, registration, transform application, filling, and remodelling outputs.
- MotionScoreHRpQCT module: run prediction/review workflows from the MotionScore core package and export grading tables.
- Scanco I/O module: read and export AIM images, import/export manufacturer transform DAT files, and expose transform registry status where relevant.
- Contour and mask tools module: interactive review and simple generation/export utilities for HR-pQCT masks and segmentations.
- Shared toolbox utilities: dependency installation, environment diagnostics, dataset root selection, progress logging, volume loading, table export, and common UI widgets.

## Integration Contract

- Core packages own persistent dataset artifacts and registries.
- Slicer modules should write through public core APIs or CLIs.
- Shared Slicer utilities may understand core artifact paths, but should not create alternate sidecar state for the same concepts.
- The transform convention exposed to Slicer remains SimpleITK/LPS physical coordinates, moving -> fixed.

## Migration Path

1. Keep current Slicer extensions releasable while core interoperability and storage work lands.
2. Scaffold the umbrella toolbox with shared utilities and one migrated TimelapsedHRpQCT module.
3. Add MotionScoreHRpQCT as a second module once shared installer/logging patterns are stable.
4. Add Scanco I/O and contour/mask tools as small modules that call core APIs.
5. Deprecate standalone Slicer extensions only after the toolbox has matching functionality and release packaging.
