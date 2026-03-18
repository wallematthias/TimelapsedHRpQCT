from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import SimpleITK as sitk


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from timelapsedhrpqct.processing.contour_generation import (  # noqa: E402
    ContourGenerationParams,
    generate_masks_from_image,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate full/trab/cort/seg masks for a single image file.",
    )
    parser.add_argument("image_path", type=Path, help="Path to an input image, typically .mha.")
    parser.add_argument(
        "--site",
        choices=("radius", "tibia", "knee", "misc"),
        default="radius",
        help="Site preset to use for inner contour defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to the image directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output stem. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--no-seg",
        action="store_true",
        help="Skip writing the segmentation output.",
    )
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print mask-generation metadata as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and stage durations while generating masks.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    image_path = args.image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    output_dir = (args.output_dir.resolve() if args.output_dir is not None else image_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem or image_path.stem

    started = time.perf_counter()
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    if args.verbose:
        print(f"[mask-test] read image: {time.perf_counter() - started:.3f}s")

    params = ContourGenerationParams()
    params.inner.site = args.site

    started = time.perf_counter()
    result = generate_masks_from_image(image=image, params=params, verbose=args.verbose)
    if args.verbose:
        print(f"[mask-test] generate masks total: {time.perf_counter() - started:.3f}s")

    full_path = output_dir / f"{stem}_mask-full.mha"
    trab_path = output_dir / f"{stem}_mask-trab.mha"
    cort_path = output_dir / f"{stem}_mask-cort.mha"
    seg_path = output_dir / f"{stem}_seg.mha"

    started = time.perf_counter()
    sitk.WriteImage(result.full, str(full_path))
    sitk.WriteImage(result.trab, str(trab_path))
    sitk.WriteImage(result.cort, str(cort_path))
    if not args.no_seg:
        sitk.WriteImage(result.seg, str(seg_path))
    if args.verbose:
        print(f"[mask-test] write outputs: {time.perf_counter() - started:.3f}s")

    print(f"wrote: {full_path}")
    print(f"wrote: {trab_path}")
    print(f"wrote: {cort_path}")
    if not args.no_seg:
        print(f"wrote: {seg_path}")

    if args.print_metadata:
        print(json.dumps(result.metadata, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
