from __future__ import annotations

import argparse
import faulthandler
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from timelapsedhrpqct.config.loader import load_config
from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_filled_sessions_by_subject_site,
    group_fused_sessions_by_subject_site,
    group_imported_stacks_by_subject_site_and_stack,
    iter_filled_session_records,
    iter_fused_session_records,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_visualize_dir,
    analysis_metadata_path,
    filled_image_path,
    filled_seg_path,
    filling_metadata_path,
    final_transform_path,
    timelapse_baseline_transform_path,
)
from timelapsedhrpqct.dataset.discovery import discover_raw_sessions
from timelapsedhrpqct.dataset.layout import (
    get_derivative_session_dir,
    get_derivatives_root,
    get_sourcedata_session_dir,
)
from timelapsedhrpqct.processing.stacks import compute_stack_ranges

def _resolve_default_config_path() -> Path:
    package_default = Path(__file__).resolve().parent / "configs" / "defaults.yml"
    repo_default = Path(__file__).resolve().parents[2] / "configs" / "defaults.yml"
    if package_default.is_file():
        return package_default
    if repo_default.is_file():
        return repo_default
    return package_default


DEFAULT_CONFIG_PATH = _resolve_default_config_path()


def _print_citation_notice() -> None:
    print("[timelapse] Please cite when using this pipeline:")
    print(
        "[timelapse]   Walle M et al. Bone. 2023;172:116780."
    )
    print(
        "[timelapse]   Whittier DE et al. Bone. 2023;176:116893."
    )
    print(
        "[timelapse]   See README.md for the full citation list."
    )
    print()


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to YAML configuration file. "
            f"Defaults to {DEFAULT_CONFIG_PATH}."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timelapse",
        description=(
            "MultistackRegistration: longitudinal HR-pQCT import, mask/seg generation, "
            "registration, optional multistack correction, transform application, "
            "optional filling, and analysis."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------
    import_parser = subparsers.add_parser(
        "import",
        help="Import raw AIM sessions and split them into per-stack artifacts.",
    )
    import_parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory containing raw AIM files.",
    )
    import_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output dataset root. Defaults to <input_root>/imported_dataset "
            "if not provided."
        ),
    )
    _add_config_argument(import_parser)
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be imported without writing any files.",
    )

    # ------------------------------------------------------------------
    # generate-masks
    # ------------------------------------------------------------------
    gm_parser = subparsers.add_parser(
        "generate-masks",
        help=(
            "Generate missing or full stack-level masks/segmentation after import. "
            "This runs on imported stack images before registration/transforms."
        ),
    )
    gm_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing imported stack artifacts.",
    )
    _add_config_argument(gm_parser)

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------
    tl_parser = subparsers.add_parser(
        "register",
        help=(
            "Estimate stack-wise longitudinal transforms and compose them into "
            "baseline-space transforms."
        ),
    )
    tl_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing imported stack artifacts.",
    )
    _add_config_argument(tl_parser)

    # ------------------------------------------------------------------
    # stackcorrect
    # ------------------------------------------------------------------
    sc_parser = subparsers.add_parser(
        "stackcorrect",
        help=(
            "Estimate multistack correction transforms, write stack-correction "
            "QC products, and write canonical final transforms."
        ),
    )
    sc_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing imported stack artifacts and timelapse transforms.",
    )
    _add_config_argument(sc_parser)

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    at_parser = subparsers.add_parser(
        "transform",
        help=(
            "Apply canonical final transforms once to original grayscale stacks "
            "and masks, and write fused transformed outputs."
        ),
    )
    at_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing imported stack artifacts and final transforms.",
    )
    _add_config_argument(at_parser)

    # ------------------------------------------------------------------
    # fill
    # ------------------------------------------------------------------
    fill_parser = subparsers.add_parser(
        "fill",
        help="Fill missing regions in fused transformed images.",
    )
    fill_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing transformed fused images.",
    )
    _add_config_argument(fill_parser)

    # ------------------------------------------------------------------
    # analyse
    # ------------------------------------------------------------------
    analyse_parser = subparsers.add_parser(
        "analyse",
        help="Run downstream remodelling analysis and trajectory analysis.",
    )
    analyse_parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing transformed / filled outputs.",
    )
    _add_config_argument(analyse_parser)
    analyse_parser.add_argument(
        "--thr",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional remodeling threshold override(s). "
            "Example: --thr 225 250 275"
        ),
    )
    analyse_parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional cluster size override(s). "
            "Example: --clusters 5 12 18"
        ),
    )
    analyse_parser.add_argument(
        "--visualize",
        type=float,
        nargs=2,
        metavar=("THR", "CLUSTER"),
        default=None,
        help=(
            "Optionally save remodelling label images for one threshold / cluster "
            "combination. Example: --visualize 250 12"
        ),
    )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run the pipeline in sequence.",
    )
    run_parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory containing raw AIM files.",
    )
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output dataset root. Defaults to <input_root>/imported_dataset "
            "if not provided."
        ),
    )
    _add_config_argument(run_parser)
    run_parser.add_argument(
        "--mode",
        choices=("regular", "multistack"),
        default="regular",
        help=(
            "Pipeline mode. "
            "'regular' skips stack correction and filling. "
            "'multistack' runs the full multistack workflow."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the import stage without writing any files.",
    )
    run_parser.add_argument(
        "--thr",
        type=float,
        nargs="+",
        default=None,
        help="Optional analysis remodeling threshold override(s).",
    )
    run_parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
        help="Optional analysis cluster size override(s).",
    )
    run_parser.add_argument(
        "--visualize",
        type=float,
        nargs=2,
        metavar=("THR", "CLUSTER"),
        default=None,
        help="Optional analysis visualization threshold / cluster pair.",
    )

    return parser


def _default_output_root(input_root: Path) -> Path:
    return input_root / "imported_dataset"


def _load_config_or_die(config_path: Path) -> AppConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_config(config_path)


def _print_session_preview(
    session,
    output_root: Path,
    config: AppConfig,
) -> None:
    from timelapsedhrpqct.io.aim import read_aim

    print(f"Subject: {session.subject_id}")
    print(f"  Session: {session.session_id}")
    print(f"    image : {session.raw_image_path.name}")

    for role in ("cort", "trab", "full"):
        if role in session.raw_mask_paths:
            print(f"    {role:5} : {session.raw_mask_paths[role].name}")

    if session.raw_seg_path is not None:
        print(f"    seg   : {session.raw_seg_path.name}")

    try:
        image, _meta = read_aim(session.raw_image_path)
        z_slices = image.GetSize()[2]
        stack_ranges = compute_stack_ranges(
            z_slices=z_slices,
            stack_depth=config.import_.stack_depth,
            on_incomplete_stack=config.import_.on_incomplete_stack,
        )
        print(
            f"    stacks: {len(stack_ranges)} "
            f"(stack_depth={config.import_.stack_depth}, z={z_slices})"
        )
    except Exception as exc:
        print(f"    stacks: <unable to inspect: {exc}>")

    sourcedata_dir = get_sourcedata_session_dir(output_root, session)
    derivatives_dir = get_derivative_session_dir(output_root, session)
    print(f"    sourcedata : {sourcedata_dir}")
    print(f"    derivatives: {derivatives_dir}")
    print()


def _expected_stack_count_for_session(session, config: AppConfig) -> int:
    from timelapsedhrpqct.io.aim import read_aim

    image, _meta = read_aim(session.raw_image_path)
    z_slices = image.GetSize()[2]
    stack_ranges = compute_stack_ranges(
        z_slices=z_slices,
        stack_depth=config.import_.stack_depth,
        on_incomplete_stack=config.import_.on_incomplete_stack,
    )
    return len(stack_ranges)


def _sessions_needing_import(sessions, dataset_root: Path, config: AppConfig) -> list:
    existing = defaultdict(set)
    for record in iter_imported_stack_records(dataset_root):
        existing[(record.subject_id, record.session_id)].add(int(record.stack_index))

    needed = []
    for session in sessions:
        expected_count = _expected_stack_count_for_session(session, config)
        imported_indices = existing.get((session.subject_id, session.session_id), set())
        if len(imported_indices) < expected_count:
            needed.append(session)
    return needed


def _needs_mask_generation(dataset_root: Path) -> bool:
    for record in iter_imported_stack_records(dataset_root):
        if not record.mask_paths.get("full", Path()).exists():
            return True
        if not record.mask_paths.get("trab", Path()).exists():
            return True
        if not record.mask_paths.get("cort", Path()).exists():
            return True
        if record.seg_path is None or not record.seg_path.exists():
            return True
    return False


def _needs_timelapse_registration(dataset_root: Path) -> bool:
    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_site_and_stack(records)
    for (subject_id, site), stacks_by_index in grouped.items():
        for stack_index, stack_records in stacks_by_index.items():
            baseline_session = stack_records[0].session_id
            for record in stack_records:
                path = timelapse_baseline_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    moving_session=record.session_id,
                    baseline_session=baseline_session,
                )
                if path.exists():
                    continue
                legacy_path = timelapse_baseline_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stack_index=stack_index,
                    moving_session=record.session_id,
                    baseline_session=baseline_session,
                )
                if not legacy_path.exists():
                    return True
    return False


def _needs_stack_correction(dataset_root: Path) -> bool:
    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_site_and_stack(records)
    for (subject_id, site), stacks_by_index in grouped.items():
        for stack_index, stack_records in stacks_by_index.items():
            baseline_session = stack_records[0].session_id
            for record in stack_records:
                path = final_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    moving_session=record.session_id,
                    baseline_session=baseline_session,
                )
                if path.exists():
                    continue
                legacy_path = final_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stack_index=stack_index,
                    moving_session=record.session_id,
                    baseline_session=baseline_session,
                )
                if not legacy_path.exists():
                    return True
    return False


def _needs_apply_transforms(dataset_root: Path) -> bool:
    records = iter_fused_session_records(dataset_root)
    if not records:
        return True
    for record in records:
        if not record.image_path.exists():
            return True
        if record.metadata_path is None or not record.metadata_path.exists():
            return True
        full_mask = record.mask_paths.get("full")
        if full_mask is None or not full_mask.exists():
            return True
    return False


def _needs_filling(dataset_root: Path) -> bool:
    fused_by_subject = group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root))
    filled_by_subject = group_filled_sessions_by_subject_site(iter_filled_session_records(dataset_root))
    if not fused_by_subject:
        return False
    for subject_site, fused_records in fused_by_subject.items():
        filled_records = {record.session_id: record for record in filled_by_subject.get(subject_site, [])}
        for fused in fused_records:
            filled = filled_records.get(fused.session_id)
            if filled is None:
                return True
            if not filled.image_path.exists():
                return True
            if not filled.full_mask_path.exists():
                return True
            if not filled.filladded_mask_path.exists():
                return True
            if filled.metadata_path is None or not filled.metadata_path.exists():
                return True
    return False


def _requested_analysis_settings(
    config: AppConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    analysis_cfg = getattr(config, "analysis", None)
    vis_cfg = getattr(config, "visualization", None)

    thresholds = [225.0]
    cluster_sizes = [12]
    pair_mode = "adjacent"
    erosion_voxels = 1
    use_filled_images = False
    compartments = ["trab", "cort", "full"]
    space = "baseline_common"
    gaussian_filter = True
    gaussian_sigma = 1.2

    if analysis_cfg is not None:
        space = str(getattr(analysis_cfg, "space", space))
        thresholds = [float(x) for x in getattr(analysis_cfg, "thresholds", thresholds)]
        cluster_sizes = [int(x) for x in getattr(analysis_cfg, "cluster_sizes", cluster_sizes)]
        pair_mode = str(getattr(analysis_cfg, "pair_mode", pair_mode))
        use_filled_images = bool(getattr(analysis_cfg, "use_filled_images", use_filled_images))
        compartments = list(getattr(analysis_cfg, "compartments", compartments))
        gaussian_filter = bool(getattr(analysis_cfg, "gaussian_filter", gaussian_filter))
        gaussian_sigma = float(getattr(analysis_cfg, "gaussian_sigma", gaussian_sigma))
        valid_region_cfg = getattr(analysis_cfg, "valid_region", None)
        if valid_region_cfg is not None:
            erosion_voxels = int(getattr(valid_region_cfg, "erosion_voxels", erosion_voxels))

    visualization_enabled = False
    visualization_threshold: float | None = None
    visualization_cluster_size: int | None = None
    if vis_cfg is not None:
        visualization_enabled = bool(getattr(vis_cfg, "enabled", False))
        raw_threshold = getattr(vis_cfg, "threshold", None)
        raw_cluster_size = getattr(vis_cfg, "cluster_size", None)
        visualization_threshold = float(raw_threshold) if raw_threshold is not None else None
        visualization_cluster_size = (
            int(raw_cluster_size) if raw_cluster_size is not None else None
        )

    if args.thr is not None:
        thresholds = [float(x) for x in args.thr]
    if args.clusters is not None:
        cluster_sizes = [int(x) for x in args.clusters]
    if args.visualize is not None:
        visualization_enabled = True
        visualization_threshold = float(args.visualize[0])
        visualization_cluster_size = int(args.visualize[1])

    visualization_requested = (
        visualization_enabled
        and visualization_threshold is not None
        and visualization_cluster_size is not None
    )

    return {
        "method": str(getattr(analysis_cfg, "method", "grayscale_and_binary"))
        if analysis_cfg is not None
        else "grayscale_and_binary",
        "space": space,
        "use_filled_images": use_filled_images,
        "compartments": compartments,
        "thresholds": thresholds,
        "cluster_sizes": cluster_sizes,
        "pair_mode": pair_mode,
        "erosion_voxels": erosion_voxels,
        "gaussian_filter": gaussian_filter,
        "gaussian_sigma": gaussian_sigma,
        "visualization_enabled": visualization_requested,
        "visualization_threshold": visualization_threshold if visualization_requested else None,
        "visualization_cluster_size": (
            visualization_cluster_size if visualization_requested else None
        ),
    }


def _needs_analysis(
    dataset_root: Path,
    config: AppConfig,
    args: argparse.Namespace,
) -> bool:
    def _matches_requested_setting(payload: dict, key: str, value: object) -> bool:
        if key not in payload:
            return True
        return payload.get(key) == value

    def _analysis_output_exists(path_str: str) -> bool:
        path = Path(path_str)
        if not path.is_absolute():
            path = dataset_root / path
        return path.exists()

    if args.thr is not None or args.clusters is not None or args.visualize is not None:
        return True
    requested = _requested_analysis_settings(config, args)
    fused_by_subject = group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root))
    if not fused_by_subject:
        return False
    for (subject_id, site), sessions in fused_by_subject.items():
        if len(sessions) < 2:
            continue
        meta_path = analysis_metadata_path(dataset_root, subject_id, site)
        if not meta_path.exists():
            legacy_meta_path = analysis_metadata_path(dataset_root, subject_id)
            if legacy_meta_path.exists():
                meta_path = legacy_meta_path
            else:
                return True
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        for key, value in requested.items():
            if not _matches_requested_setting(payload, key, value):
                return True
        for key in ("pairwise_csv", "trajectory_csv"):
            out = payload.get(key)
            if not out or not _analysis_output_exists(out):
                return True
        if requested["visualization_enabled"]:
            visualize_dir = analysis_visualize_dir(dataset_root, subject_id, site)
            if not visualize_dir.exists() or not any(visualize_dir.glob("*.mha")):
                return True
    return False


def _filling_enabled(config: AppConfig) -> bool:
    fusion_cfg = getattr(config, "fusion", None)
    if fusion_cfg is None:
        return True
    return bool(getattr(fusion_cfg, "enable_filling", True))


def _multistack_correction_enabled(config: AppConfig) -> bool:
    cfg = getattr(config, "multistack_correction", None)
    if cfg is None:
        return True
    return bool(getattr(cfg, "enabled", True))


def _cmd_import(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.import_aim import import_subject_sessions

    config = _load_config_or_die(args.config)

    input_root: Path = args.input_root.resolve()
    output_root: Path = (args.output_root or _default_output_root(input_root)).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    sessions = discover_raw_sessions(
        root=input_root,
        discovery_config=config.discovery,
    )

    if not sessions:
        print(f"[timelapse] No raw sessions found under: {input_root}")
        return 0

    if args.dry_run:
        print("[timelapse] DRY RUN")
        print()
        print(f"[timelapse] Found {len(sessions)} raw session(s)")
        print(f"[timelapse] Output root: {output_root}")
        print()

        for session in sessions:
            _print_session_preview(
                session=session,
                output_root=output_root,
                config=config,
            )

        print(f"[timelapse] {len(sessions)} session(s) would be imported.")
        return 0

    sessions = _sessions_needing_import(
        sessions=sessions,
        dataset_root=output_root,
        config=config,
    )
    if not sessions:
        print("[timelapse] All discovered raw sessions already imported -> skip")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[timelapse] Found {len(sessions)} raw session(s)")
    print(f"[timelapse] Output root: {output_root}")

    sessions_by_subject: dict[str, list] = {}
    for session in sessions:
        sessions_by_subject.setdefault(session.subject_id, []).append(session)

    total_stacks = 0

    for subject_id, subject_sessions in sorted(sessions_by_subject.items()):
        subject_sessions = sorted(subject_sessions, key=lambda s: s.session_id)

        print(
            f"[timelapse] Importing subject sub-{subject_id} "
            f"with {len(subject_sessions)} session(s)"
        )

        subject_artifacts = import_subject_sessions(
            raw_sessions=subject_sessions,
            output_root=output_root,
            config=config,
        )
        total_stacks += len(subject_artifacts)

        print(
            f"[timelapse]   -> wrote {len(subject_artifacts)} stack artifact(s) "
            f"for sub-{subject_id}"
        )

    print("[timelapse] Import complete")
    print(f"[timelapse] Total stacks written: {total_stacks}")
    return 0


def _cmd_generate_masks(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.generate_masks import run_mask_generation

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_mask_generation(
        dataset_root=dataset_root,
        config=config,
    )
    return 0


def _cmd_timelapse_register(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.timelapse_registration import (
        run_timelapse_registration,
    )

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_timelapse_registration(
        dataset_root=dataset_root,
        config=config,
    )
    return 0


def _cmd_stack_correct(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.multistack_correction import (
        run_stack_correction,
    )

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_stack_correction(
        dataset_root=dataset_root,
        config=config,
    )
    return 0


def _cmd_apply_transforms(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.apply_transforms import run_apply_transforms

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_apply_transforms(
        dataset_root=dataset_root,
        config=config,
    )
    return 0


def _cmd_fill(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.filling import run_filling

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    run_filling(
        dataset_root=dataset_root,
        config=config,
    )
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    from timelapsedhrpqct.workflows.analysis import run_analysis

    config = _load_config_or_die(args.config)
    dataset_root: Path = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    visualize_pair = None
    if args.visualize is not None:
        visualize_pair = (float(args.visualize[0]), int(args.visualize[1]))

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=args.thr,
        clusters=args.clusters,
        visualize=visualize_pair,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    input_root: Path = args.input_root.resolve()
    output_root: Path = (args.output_root or _default_output_root(input_root)).resolve()
    config = _load_config_or_die(args.config)

    _print_citation_notice()

    # 1. import
    discovered_sessions = discover_raw_sessions(
        root=input_root,
        discovery_config=config.discovery,
    )
    if not discovered_sessions:
        print(f"[timelapse] No raw sessions found under: {input_root}")
        return 0

    import_args = argparse.Namespace(
        input_root=input_root,
        output_root=output_root,
        config=args.config,
        dry_run=args.dry_run,
    )
    rc = _cmd_import(import_args)
    if rc != 0 or args.dry_run:
        return rc

    dataset_root = output_root

    # 2. generate masks / seg where needed
    if _needs_mask_generation(dataset_root):
        gm_args = argparse.Namespace(
            dataset_root=dataset_root,
            config=args.config,
        )
        rc = _cmd_generate_masks(gm_args)
        if rc != 0:
            return rc
    else:
        print("[timelapse] generate-masks: all imported stacks already complete -> skip")

    # 3. register
    if _needs_timelapse_registration(dataset_root):
        tl_args = argparse.Namespace(
            dataset_root=dataset_root,
            config=args.config,
        )
        rc = _cmd_timelapse_register(tl_args)
        if rc != 0:
            return rc
    else:
        print("[timelapse] register: baseline transforms already exist -> skip")

    if args.mode == "multistack" and _multistack_correction_enabled(config):
        # 4. stackcorrect
        if _needs_stack_correction(dataset_root):
            sc_args = argparse.Namespace(
                dataset_root=dataset_root,
                config=args.config,
            )
            rc = _cmd_stack_correct(sc_args)
            if rc != 0:
                return rc
        else:
            print("[timelapse] stackcorrect: final transforms already exist -> skip")
    elif args.mode == "multistack":
        print("[timelapse] stackcorrect: disabled by config.multistack_correction.enabled -> skip")

    # 5. transform
    if _needs_apply_transforms(dataset_root):
        at_args = argparse.Namespace(
            dataset_root=dataset_root,
            config=args.config,
        )
        rc = _cmd_apply_transforms(at_args)
        if rc != 0:
            return rc
    else:
        print("[timelapse] transform: fused transformed sessions already exist -> skip")

    if args.mode == "multistack":
        # 6. fill
        if _filling_enabled(config):
            if _needs_filling(dataset_root):
                fill_args = argparse.Namespace(
                    dataset_root=dataset_root,
                    config=args.config,
                )
                rc = _cmd_fill(fill_args)
                if rc != 0:
                    return rc
            else:
                print("[timelapse] fill: filled sessions already exist -> skip")
        else:
            print("[timelapse] fill: disabled by config.fusion.enable_filling -> skip")

    # 7. analyse
    if not _filling_enabled(config) and bool(getattr(config.analysis, "use_filled_images", False)):
        raise ValueError(
            "analysis.use_filled_images=true requires fusion.enable_filling=true "
            "or a prior fill stage with existing filled outputs."
        )

    if _needs_analysis(dataset_root, config, args):
        analyse_args = argparse.Namespace(
            dataset_root=dataset_root,
            config=args.config,
            thr=args.thr,
            clusters=args.clusters,
            visualize=args.visualize,
        )
        rc = _cmd_analyse(analyse_args)
        if rc != 0:
            return rc
    else:
        print("[timelapse] analyse: analysis outputs already exist -> skip")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    enable_faulthandler = os.environ.get("TIMELAPSE_FAULTHANDLER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_faulthandler:
        faulthandler.enable(all_threads=True)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "import":
        return _cmd_import(args)
    if args.command == "generate-masks":
        return _cmd_generate_masks(args)
    if args.command == "register":
        return _cmd_timelapse_register(args)
    if args.command == "stackcorrect":
        return _cmd_stack_correct(args)
    if args.command == "transform":
        return _cmd_apply_transforms(args)
    if args.command == "fill":
        return _cmd_fill(args)
    if args.command == "analyse":
        return _cmd_analyse(args)
    if args.command == "run":
        return _cmd_run(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
