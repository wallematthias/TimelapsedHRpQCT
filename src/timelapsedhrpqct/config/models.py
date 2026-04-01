from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ImportConfig:
    stack_depth: int = 168
    on_incomplete_stack: str = "keep_last"

    crop_to_subject_box: bool = True
    crop_threshold_bmd: float = 450.0
    crop_padding_voxels: int = 5
    crop_num_largest_components: int = 1


@dataclass(slots=True)
class DiscoveryConfig:
    """
    Controls filename-based dataset discovery.

    If `session_regex` is provided, it is used to extract subject/session/role
    from filenames. Named groups supported:
    - subject
    - session
    - role (optional)
    """
    session_regex: str | None = None
    default_site: str = "radius"
    site_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "radius": ["DR", "RADIUS", "RAD"],
            "tibia": ["DT", "TIBIA", "TIB"],
            "knee": ["KN", "KNEE"],
            "radius_left": ["RL", "RADIUS_LEFT", "RADL", "LEFT_RADIUS"],
            "radius_right": ["RR", "RADIUS_RIGHT", "RADR", "RIGHT_RADIUS"],
            "tibia_left": ["TL", "TIBIA_LEFT", "TIBL", "LEFT_TIBIA"],
            "tibia_right": ["TR", "TIBIA_RIGHT", "TIBR", "RIGHT_TIBIA"],
            "knee_left": ["KL", "KNL", "KNEE_LEFT", "KNEEL", "LEFT_KNEE"],
            "knee_right": ["KR", "KNR", "KNEE_RIGHT", "KNEER", "RIGHT_KNEE"],
        }
    )
    session_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "T1": ["BASELINE", "BL"],
            "T2": ["FOLLOWUP", "FOLLOWUP1", "FL", "FU"],
        }
    )
    role_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "cort": ["CORT_MASK", "_CORT", "CORTICAL"],
            "trab": ["TRAB_MASK", "_TRAB", "TRABECULAR"],
            "full": ["FULL_MASK", "_FULL"],
            "regmask": ["REGMASK", "_REGMASK", "_REG"],
            "seg": ["_SEG", "SEG"],
        }
    )


@dataclass(slots=True)
class OuterContourConfig:
    periosteal_threshold: float = 300.0
    periosteal_kernelsize: int = 5
    periosteal_open_radius: int = 2
    morphology_downsample_factor: int = 1
    morphology_refine_edges: bool = False
    morphology_refine_band_voxels: int = 3
    gaussian_sigma: float = 1.5
    gaussian_truncate: float = 1.0
    expansion_depth: list[int] = field(default_factory=lambda: [0, 5])
    init_pad: int = 15
    fill_holes: bool = True
    use_adaptive_threshold: bool = True


@dataclass(slots=True)
class InnerContourConfig:
    site: str = "misc"
    endosteal_threshold: float = 500.0
    endosteal_kernelsize: int = 3
    morphology_downsample_factor: int = 1
    morphology_refine_edges: bool = False
    morphology_refine_band_voxels: int = 3
    gaussian_sigma: float = 1.5
    gaussian_truncate: float = 1.0
    peel: int = 3
    expansion_depth: list[int] = field(default_factory=lambda: [0, 3, 10, 3])
    trabecular_close_radius: int | None = None
    init_pad: int = 30
    use_adaptive_threshold: bool = False


@dataclass(slots=True)
class MaskSegmentationConfig:
    enabled: bool = True
    method: str = "global"  # "global" | "adaptive"
    gaussian_sigma: float = 0.8
    trab_threshold: float = 320.0
    cort_threshold: float = 450.0
    adaptive_low_threshold: float = 190.0
    adaptive_high_threshold: float = 450.0
    adaptive_block_size: int = 13
    min_size_voxels: int = 64
    keep_largest_component: bool = True


@dataclass(slots=True)
class MasksConfig:
    generate: bool = False
    overwrite: bool = False
    roles: list[str] = field(default_factory=lambda: ["full", "trab", "cort"])
    generate_segmentation: bool = True
    site_selection: dict[str, object] = field(
        default_factory=lambda: {
            "default_site": "radius",
            "patterns": {
                "radius": ["radius", "rad"],
                "tibia": ["tibia", "tib"],
                "knee": ["knee"],
            },
        }
    )
    site_defaults: dict[str, dict[str, dict[str, object]]] = field(
        default_factory=lambda: {
            "radius": {
                "inner": {
                    "endosteal_threshold": 450,
                    "endosteal_kernelsize": 3,
                    "morphology_downsample_factor": 1,
                    "gaussian_sigma": 1.5,
                    "gaussian_truncate": 1.0,
                    "peel": 3,
                    "trabecular_close_radius": 15,
                    "init_pad": 30,
                    "use_adaptive_threshold": False,
                }
            },
            "tibia": {
                "inner": {
                    "endosteal_threshold": 450,
                    "endosteal_kernelsize": 3,
                    "morphology_downsample_factor": 1,
                    "gaussian_sigma": 1.5,
                    "gaussian_truncate": 1.0,
                    "peel": 3,
                    "trabecular_close_radius": 25,
                    "init_pad": 30,
                    "use_adaptive_threshold": False,
                }
            },
            "knee": {
                "inner": {
                    "endosteal_threshold": 450,
                    "endosteal_kernelsize": 3,
                    "morphology_downsample_factor": 1,
                    "gaussian_sigma": 1.5,
                    "gaussian_truncate": 1.0,
                    "peel": 3,
                    "trabecular_close_radius": 25,
                    "init_pad": 30,
                    "use_adaptive_threshold": False,
                }
            },
        }
    )

    outer: OuterContourConfig = field(default_factory=OuterContourConfig)
    inner: InnerContourConfig = field(default_factory=InnerContourConfig)
    segmentation: MaskSegmentationConfig = field(default_factory=MaskSegmentationConfig)


@dataclass(slots=True)
class TimelapsedRegistrationConfig:
    strategy: str = "sequential_to_baseline"
    reference_session: str = "baseline"
    transform_type: str = "euler"
    metric: str = "correlation"
    sampling_percentage: float = 0.002
    interpolator: str = "linear"
    optimizer: str = "adaptive_stochastic_gradient_descent"
    number_of_iterations: int = 250
    automatic_parameter_estimation: bool = True
    sp_a: float = 20.0
    maximum_step_length: float | None = None
    sigmoid_scale_factor: float = 0.1
    number_of_gradient_measurements: int = 0
    number_of_jacobian_measurements: int = 1000

    initializer: str = "geometry"
    number_of_resolutions: int = 6
    use_masks: bool = True

    debug: bool = False


@dataclass(slots=True)
class MultistackCorrectionConfig:
    enabled: bool = True
    method: str = "superstack"
    overlap_crop_buffer_voxels: int = 40

    transform_type: str = "euler"
    metric: str = "correlation"
    sampling_percentage: float = 0.002
    interpolator: str = "linear"
    optimizer: str = "adaptive_stochastic_gradient_descent"
    number_of_iterations: int = 250
    automatic_parameter_estimation: bool = True
    sp_a: float = 20.0
    maximum_step_length: float | None = None
    sigmoid_scale_factor: float = 0.1
    number_of_gradient_measurements: int = 0
    number_of_jacobian_measurements: int = 1000
    initial_translation_voxels: list[float] = field(
        default_factory=lambda: [0.0, 0.0, -20.0]
    )

    initializer: str = "geometry"
    number_of_resolutions: int = 4
    use_masks: bool = True

    debug: bool = False


@dataclass(slots=True)
class TransformConfig:
    image_interpolator: str = "linear"
    mask_interpolator: str = "nearest"


@dataclass(slots=True)
class FusionConfig:
    save_fused: bool = True
    save_fusedfilled: bool = False
    enable_filling: bool = False


@dataclass(slots=True)
class AnalysisValidRegionConfig:
    erosion_voxels: int = 1


@dataclass(slots=True)
class AnalysisConfig:
    space: str = "pairwise_fixed_t0"
    method: str = "grayscale_and_binary"
    compartments: list[str] = field(default_factory=lambda: ["trab", "cort", "full"])
    thresholds: list[float] = field(default_factory=lambda: [225.0])
    cluster_sizes: list[int] = field(default_factory=lambda: [12])
    pair_mode: str = "adjacent"
    use_filled_images: bool = False
    gaussian_filter: bool = True
    gaussian_sigma: float = 1.2
    valid_region: AnalysisValidRegionConfig = field(default_factory=AnalysisValidRegionConfig)


@dataclass(slots=True)
class VisualizationLabelMapConfig:
    resorption: int = 1
    demineralisation: int = 2
    quiescent: int = 2
    formation: int = 3
    mineralisation: int = 2


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    threshold: float | None = 225.0
    cluster_size: int | None = 12
    label_map: VisualizationLabelMapConfig = field(default_factory=VisualizationLabelMapConfig)


@dataclass(slots=True)
class AppConfig:
    import_: ImportConfig = field(default_factory=ImportConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    masks: MasksConfig = field(default_factory=MasksConfig)
    timelapsed_registration: TimelapsedRegistrationConfig = field(
        default_factory=TimelapsedRegistrationConfig
    )
    multistack_correction: MultistackCorrectionConfig = field(
        default_factory=MultistackCorrectionConfig
    )
    transform: TransformConfig = field(default_factory=TransformConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
