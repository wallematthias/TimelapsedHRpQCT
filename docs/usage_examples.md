# Usage Examples

## Example 1: Dry-run a new dataset

```bash
timelapse import /data/hrpqct --dry-run
```

Use this first to verify:

- sessions are discovered correctly
- masks and segmentations are associated with the right sessions
- stack splitting looks plausible

## Example 2: Full multistack run

```bash
timelapse run /data/hrpqct --mode multistack
```

This executes:

1. import
2. generate masks
3. timelapsed registration
4. multistack correction
5. transform application
6. filling
7. analysis

## Example 3: Resume after a previous run

```bash
timelapse run /data/hrpqct --mode multistack
```

Rerunning the same command should now skip completed stages based on existing artifacts and output files.

## Example 4: Re-run analysis with different thresholds

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

## Example 5: Single-stack workflow

```bash
timelapse run /data/hrpqct --config configs/example_single_stack.yml --mode regular
```

Use this when each session already contains one complete stack and you do not need multistack correction or filling.

Pass `--config /path/to/other.yml` when you want to use a non-default configuration file.
