from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from multistack_registration.config.models import (
    AnalysisConfig,
    AnalysisValidRegionConfig,
    AppConfig,
    DiscoveryConfig,
    FusionConfig,
    ImportConfig,
    InnerContourConfig,
    MaskSegmentationConfig,
    MasksConfig,
    MultistackCorrectionConfig,
    OuterContourConfig,
    TimelapsedRegistrationConfig,
    TransformConfig,
    VisualizationConfig,
    VisualizationLabelMapConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in allowed}


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    raw = _read_yaml(cfg_path)

    import_raw = raw.get("import", {})
    discovery_raw = raw.get("discovery", {})
    masks_raw = raw.get("masks", {})
    timelapsed_raw = raw.get("timelapsed_registration", {})
    multistack_raw = raw.get("multistack_correction", {})
    transform_raw = raw.get("transform", {})
    fusion_raw = raw.get("fusion", {})
    analysis_raw = raw.get("analysis", {})
    visualization_raw = raw.get("visualization", {})

    masks_outer_raw = masks_raw.get("outer", {})
    masks_inner_raw = masks_raw.get("inner", {})
    masks_seg_raw = masks_raw.get("segmentation", {})
    analysis_valid_region_raw = analysis_raw.get("valid_region", {})
    visualization_label_map_raw = visualization_raw.get("label_map", {})

    masks_cfg = MasksConfig(
        **_filter_dataclass_kwargs(
            MasksConfig,
            {
                k: v
                for k, v in masks_raw.items()
                if k not in {"outer", "inner", "segmentation"}
            },
        ),
        outer=OuterContourConfig(
            **_filter_dataclass_kwargs(OuterContourConfig, masks_outer_raw)
        ),
        inner=InnerContourConfig(
            **_filter_dataclass_kwargs(InnerContourConfig, masks_inner_raw)
        ),
        segmentation=MaskSegmentationConfig(
            **_filter_dataclass_kwargs(MaskSegmentationConfig, masks_seg_raw)
        ),
    )

    return AppConfig(
        import_=ImportConfig(**_filter_dataclass_kwargs(ImportConfig, import_raw)),
        discovery=DiscoveryConfig(
            **_filter_dataclass_kwargs(DiscoveryConfig, discovery_raw)
        ),
        masks=masks_cfg,
        timelapsed_registration=TimelapsedRegistrationConfig(
            **_filter_dataclass_kwargs(TimelapsedRegistrationConfig, timelapsed_raw)
        ),
        multistack_correction=MultistackCorrectionConfig(
            **_filter_dataclass_kwargs(MultistackCorrectionConfig, multistack_raw)
        ),
        transform=TransformConfig(
            **_filter_dataclass_kwargs(TransformConfig, transform_raw)
        ),
        fusion=FusionConfig(**_filter_dataclass_kwargs(FusionConfig, fusion_raw)),
        analysis=AnalysisConfig(
            **_filter_dataclass_kwargs(
                AnalysisConfig,
                {k: v for k, v in analysis_raw.items() if k != "valid_region"},
            ),
            valid_region=AnalysisValidRegionConfig(
                **_filter_dataclass_kwargs(
                    AnalysisValidRegionConfig,
                    analysis_valid_region_raw,
                )
            ),
        ),
        visualization=VisualizationConfig(
            **_filter_dataclass_kwargs(
                VisualizationConfig,
                {k: v for k, v in visualization_raw.items() if k != "label_map"},
            ),
            label_map=VisualizationLabelMapConfig(
                **_filter_dataclass_kwargs(
                    VisualizationLabelMapConfig,
                    visualization_label_map_raw,
                )
            ),
        ),
    )
