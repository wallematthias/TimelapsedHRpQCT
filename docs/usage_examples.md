# Usage Examples

## Example 1: Dry-run a new dataset

```bash
timelapse import /data/hrpqct --dry-run
```

Use this first to verify:

- sessions are discovered correctly
- masks and segmentations are associated with the right sessions
- stack splitting looks plausible

## Example 2: Default pipeline run

```bash
timelapse run /data/hrpqct
```

This executes:

1. import
2. generate masks
3. timelapsed registration
4. multistack correction (only when the selected profile enables it)
5. transform application
6. filling
7. analysis

## Example 3: Standard multistack run

```bash
timelapse run /data/hrpqct --profile multistack
```

Use this when each session contains multiple stacks and you want stack correction + filling.

## Example 4: Resume after a previous run

```bash
timelapse run /data/hrpqct
```

Rerunning the same command should now skip completed stages based on existing artifacts and output files.

## Example 5: Re-run analysis with different thresholds

```bash
timelapse analyse /data/hrpqct/imported_dataset \
  --thr 225 250 275 \
  --clusters 12 18
```

Or using `run` with analysis overrides:

```bash
timelapse run /data/hrpqct \
  --profile multistack \
  --thr 225 250 \
  --clusters 12
```

When overrides are present, analysis reruns even if previous analysis outputs already exist.

## Example 6: Pediatric fracture multistack workflow

```bash
timelapse run /data/hrpqct --profile ped-fx
```

Use this for pediatric fracture/healing datasets that need multistack correction, geodesic periosteal contouring, Gaussian segmentation, and full-mask-only analysis.

For single-stack workflows, omit `--profile` or use a single-stack profile such as `standard`, `xct1-standard`, or `eth-uofc`. The default `--mode auto` follows the selected profile; use `--mode regular` or `--mode multistack` only when you need an explicit override.

Pass `--config /path/to/other.yml` when you want to use a non-default configuration file.
