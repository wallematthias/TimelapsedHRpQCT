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
4. multistack correction (only when `--mode multistack`)
5. transform application
6. filling
7. analysis

## Example 3: Full multistack run

```bash
timelapse run /data/hrpqct --mode multistack
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
  --mode multistack \
  --thr 225 250 \
  --clusters 12
```

When overrides are present, analysis reruns even if previous analysis outputs already exist.

## Example 6: Single-stack workflow

```bash
timelapse run /data/hrpqct --mode regular
```

Use this when each session already contains one complete stack and you do not need multistack correction or filling. `regular` is the default mode.

Pass `--config /path/to/other.yml` when you want to use a non-default configuration file.
