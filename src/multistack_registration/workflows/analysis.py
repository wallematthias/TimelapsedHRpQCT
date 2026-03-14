from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk

from multistack_registration.analysis.remodelling import (
    AnalysisParams,
    compute_remodelling_outputs,
)
from multistack_registration.config.models import AppConfig
from multistack_registration.dataset.derivative_paths import (
    analysis_dir,
    analysis_metadata_path,
    analysis_visualize_path,
    common_region_path,
    pairwise_remodelling_csv_path,
    trajectory_metrics_csv_path,
)
from multistack_registration.processing.analysis_io import (
    build_analysis_summary_metadata,
    discover_analysis_sessions,
    discover_analysis_subject_ids,
)
from multistack_registration.utils.sitk_helpers import (
    array_to_image,
    load_image,
    write_image,
    write_json,
    image_to_array,
)


def _free_memory() -> None:
    gc.collect()


def _get_analysis_params(config: AppConfig) -> AnalysisParams:
    cfg = getattr(config, "analysis", None)

    compartments = ["trab", "cort", "full"]
    remodeling_thresholds = [225.0]
    cluster_sizes = [12]
    pair_mode = "adjacent"
    erosion_voxels = 1
    use_filled_images = False

    if cfg is not None:
        compartments = list(getattr(cfg, "compartments", compartments))
        remodeling_thresholds = [float(x) for x in getattr(cfg, "thresholds", remodeling_thresholds)]
        cluster_sizes = [int(x) for x in getattr(cfg, "cluster_sizes", cluster_sizes)]
        pair_mode = str(getattr(cfg, "pair_mode", pair_mode))
        erosion_voxels = int(
            getattr(getattr(cfg, "valid_region", None), "erosion_voxels", erosion_voxels)
        )
        use_filled_images = bool(getattr(cfg, "use_filled_images", use_filled_images))

    vis_cfg = getattr(config, "visualization", None)
    visualize_enabled = False
    visualize_threshold: float | None = None
    visualize_cluster_size: int | None = None
    visualize_label_map = {
        "resorption": 1,
        "demineralisation": 2,
        "quiescent": 3,
        "formation": 4,
        "mineralisation": 5,
    }

    if vis_cfg is not None:
        visualize_enabled = bool(getattr(vis_cfg, "enabled", False))
        vt = getattr(vis_cfg, "threshold", None)
        vc = getattr(vis_cfg, "cluster_size", None)
        visualize_threshold = float(vt) if vt is not None else None
        visualize_cluster_size = int(vc) if vc is not None else None

        lm = getattr(vis_cfg, "label_map", None)
        if lm is not None:
            visualize_label_map = {
                "resorption": int(getattr(lm, "resorption", visualize_label_map["resorption"])),
                "demineralisation": int(
                    getattr(lm, "demineralisation", visualize_label_map["demineralisation"])
                ),
                "quiescent": int(getattr(lm, "quiescent", visualize_label_map["quiescent"])),
                "formation": int(getattr(lm, "formation", visualize_label_map["formation"])),
                "mineralisation": int(
                    getattr(lm, "mineralisation", visualize_label_map["mineralisation"])
                ),
            }

    return AnalysisParams(
        compartments=compartments,
        remodeling_thresholds=remodeling_thresholds,
        cluster_sizes=cluster_sizes,
        pair_mode=pair_mode,
        erosion_voxels=erosion_voxels,
        use_filled_images=use_filled_images,
        visualize_enabled=visualize_enabled,
        visualize_threshold=visualize_threshold,
        visualize_cluster_size=visualize_cluster_size,
        visualize_label_map=visualize_label_map,
    )


def _apply_overrides(
    params: AnalysisParams,
    thresholds: Iterable[float] | None,
    clusters: Iterable[int] | None,
    visualize: tuple[float, int] | None,
) -> AnalysisParams:
    if thresholds is not None:
        params.remodeling_thresholds = [float(x) for x in thresholds]
    if clusters is not None:
        params.cluster_sizes = [int(x) for x in clusters]
    if visualize is not None:
        params.visualize_enabled = True
        params.visualize_threshold = float(visualize[0])
        params.visualize_cluster_size = int(visualize[1])
    return params


def run_analysis(
    dataset_root: str | Path,
    config: AppConfig,
    thresholds: Iterable[float] | None = None,
    clusters: Iterable[int] | None = None,
    visualize: tuple[float, int] | None = None,
) -> None:
    dataset_root = Path(dataset_root)
    params = _apply_overrides(
        _get_analysis_params(config),
        thresholds=thresholds,
        clusters=clusters,
        visualize=visualize,
    )

    subject_ids = discover_analysis_subject_ids(dataset_root)
    if not subject_ids:
        print(f"[analysis] No subjects found under: {dataset_root}")
        return

    for subject_id in subject_ids:
        sessions = discover_analysis_sessions(
            dataset_root=dataset_root,
            subject_id=subject_id,
            use_filled_images=params.use_filled_images,
        )
        if len(sessions) < 2:
            print(f"[analysis] Skipping sub-{subject_id}: need at least 2 sessions.")
            continue

        print(
            f"[analysis] Subject sub-{subject_id}: {len(sessions)} session(s), "
            f"pair_mode={params.pair_mode}, use_filled_images={params.use_filled_images}"
        )

        ref_img = load_image(sessions[0].image_path)
        session_ids = [s.session_id for s in sessions]
        session_seg_paths = [str(s.seg_path) for s in sessions]

        image_arrs: list[np.ndarray] = []
        seg_arrs: list[np.ndarray] = []
        mask_arrs_by_role: dict[str, list[np.ndarray]] = {
            role: [] for role in ("trab", "cort", "full")
        }

        for s in sessions:
            image_arrs.append(
                image_to_array(load_image(s.image_path)).astype(np.float32, copy=False)
            )
            seg_arrs.append(
                (image_to_array(load_image(s.seg_path)) > 0).astype(bool, copy=False)
            )
            for role in ("trab", "cort", "full"):
                if role in s.mask_paths and s.mask_paths[role].exists():
                    mask_arrs_by_role[role].append(
                        image_to_array(load_image(s.mask_paths[role])) > 0
                    )
                else:
                    mask_arrs_by_role[role].append(
                        np.zeros_like(image_arrs[-1], dtype=bool)
                    )

        outputs = compute_remodelling_outputs(
            subject_id=subject_id,
            session_ids=session_ids,
            session_seg_paths=session_seg_paths,
            image_arrs=image_arrs,
            seg_arrs=seg_arrs,
            mask_arrs_by_role=mask_arrs_by_role,
            params=params,
            common_region_path_for=lambda compartment: str(
                common_region_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    compartment=compartment,
                )
            ),
        )

        for compartment, mask_arr in outputs.common_masks.items():
            common_img = array_to_image(
                mask_arr.astype(np.uint8),
                reference=ref_img,
                pixel_id=sitk.sitkUInt8,
            )
            write_image(
                common_img,
                common_region_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    compartment=compartment,
                ),
            )
            del common_img

        for (compartment, t0, t1, thr, cluster_size), label_arr in outputs.label_images.items():
            label_img = array_to_image(label_arr, reference=ref_img, pixel_id=sitk.sitkUInt8)
            write_image(
                label_img,
                analysis_visualize_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    compartment=compartment,
                    t0=t0,
                    t1=t1,
                    thr=thr,
                    cluster_size=cluster_size,
                ),
            )
            del label_img

        pairwise_df = pd.DataFrame(outputs.pairwise_rows)
        trajectory_df = pd.DataFrame(outputs.trajectory_rows)
        pairwise_path = pairwise_remodelling_csv_path(dataset_root, subject_id)
        trajectory_path = trajectory_metrics_csv_path(dataset_root, subject_id)
        pairwise_path.parent.mkdir(parents=True, exist_ok=True)
        pairwise_df.to_csv(pairwise_path, index=False)
        trajectory_df.to_csv(trajectory_path, index=False)

        analysis_meta = build_analysis_summary_metadata(
            dataset_root=dataset_root,
            subject_id=subject_id,
            use_filled_images=params.use_filled_images,
            compartments=params.compartments,
            thresholds=params.remodeling_thresholds,
            cluster_sizes=params.cluster_sizes,
            pair_mode=params.pair_mode,
            erosion_voxels=params.erosion_voxels,
            visualization_enabled=params.visualize_enabled,
            visualization_threshold=params.visualize_threshold,
            visualization_cluster_size=params.visualize_cluster_size,
            pairwise_csv=pairwise_path,
            trajectory_csv=trajectory_path,
        )
        write_json(
            analysis_meta,
            analysis_metadata_path(dataset_root, subject_id),
        )

        print(
            f"[analysis] sub-{subject_id}: wrote "
            f"{len(pairwise_df)} pairwise row(s) and {len(trajectory_df)} trajectory row(s)"
        )

        del pairwise_df, trajectory_df, image_arrs, seg_arrs, mask_arrs_by_role
        _free_memory()
