# Documentation

This documentation is split into overview material and deeper operational detail.

## Guides

- [Installation](./installation.md)
- [Usage](./usage.md)
- [Usage Examples](./usage_examples.md)
- [Annotated Defaults](./defaults_annotated.md)
- [Multistack Algorithm](./multistack_algorithm.md)
- [Timelapsed Analysis](./analysis.md)
- [Settings Reference](./settings.md)

## Recommended Reading Order

If you are new to the project:

1. Read the repository [README](/Users/matthias.walle/Documents/GitHub/TimelapsedMultistack/README.md).
2. Follow [Installation](./installation.md).
3. Read [Usage](./usage.md) and run a dry import.
4. Read [Multistack Algorithm](./multistack_algorithm.md) if you want to understand the transform logic.
5. Read [Timelapsed Analysis](./analysis.md) and [Settings Reference](./settings.md) when tuning the pipeline.

## Core Concepts

- Raw session discovery is configuration-driven.
- Imported stack artifacts are the stable handoff between ingest and downstream processing.
- Transforms always follow a moving-to-fixed convention.
- Final resampling is intended to happen once, during transform application.
- Analysis operates on fused transformed data, and can optionally use filled grayscale data.
