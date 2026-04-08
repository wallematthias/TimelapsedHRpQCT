from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

try:
    import itk
except ImportError:
    itk = None


@dataclass(slots=True)
class RegistrationSettings:
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
    initial_translation_voxels: tuple[float, float, float] = (0.0, 0.0, 0.0)
    number_of_resolutions: int = 4
    use_masks: bool = True
    debug: bool = False


@dataclass(slots=True)
class RegistrationResult:
    transform: sitk.Transform
    metric_value: float
    optimizer_stop_condition: str
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_truthy_env(value: str | None) -> bool:
    """Return whether an environment variable value enables a feature."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _image_debug_summary(image: sitk.Image) -> dict[str, Any]:
    """Build lightweight geometry/intensity summary for debug logging."""
    arr = sitk.GetArrayViewFromImage(image)
    return {
        "dimension": int(image.GetDimension()),
        "size": [int(v) for v in image.GetSize()],
        "spacing": [float(v) for v in image.GetSpacing()],
        "origin": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
    }


def _mask_debug_summary(mask: sitk.Image | None) -> dict[str, Any] | None:
    """Build mask occupancy and geometry summary for debug logging."""
    if mask is None:
        return None
    arr = sitk.GetArrayViewFromImage(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return {
        "size": [int(v) for v in mask.GetSize()],
        "spacing": [float(v) for v in mask.GetSpacing()],
        "origin": [float(v) for v in mask.GetOrigin()],
        "direction": [float(v) for v in mask.GetDirection()],
        "nonzero_voxels": int(np.count_nonzero(arr)),
        "total_voxels": int(arr.size),
        "occupancy_fraction": (float(np.count_nonzero(arr)) / float(arr.size)) if arr.size else float("nan"),
    }


def _should_trace_registration(settings: RegistrationSettings) -> bool:
    """Return whether verbose registration trace logging is enabled."""
    return bool(settings.debug) or _is_truthy_env(os.environ.get("TIMELAPSE_TRACE_REGISTRATION"))


def _build_registration_debug_context(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    settings: RegistrationSettings,
    fixed_mask: sitk.Image | None,
    moving_mask: sitk.Image | None,
    *,
    use_masks: bool,
    spatial_samples: int,
    init_transform_parameters: list[Any],
    requested_initial_translation_vox: tuple[float, float, float],
    requested_initial_translation_physical: tuple[float, float, float],
) -> dict[str, Any]:
    """Collect preflight context for registration debugging and failures."""
    return {
        "settings": {
            "transform_type": str(settings.transform_type),
            "metric": str(settings.metric),
            "optimizer": str(settings.optimizer),
            "initializer": str(settings.initializer),
            "sampling_percentage": float(settings.sampling_percentage),
            "number_of_resolutions": int(settings.number_of_resolutions),
            "number_of_iterations": int(settings.number_of_iterations),
            "automatic_parameter_estimation": bool(settings.automatic_parameter_estimation),
            "use_masks_requested": bool(settings.use_masks),
            "use_masks_effective": bool(use_masks),
            "initial_translation_voxels": [float(v) for v in requested_initial_translation_vox],
            "initial_translation_physical": [float(v) for v in requested_initial_translation_physical],
            "computed_spatial_samples": int(spatial_samples),
            "elastix_init_transform_parameters": [str(v) for v in init_transform_parameters],
        },
        "fixed_image": _image_debug_summary(fixed_image),
        "moving_image": _image_debug_summary(moving_image),
        "fixed_mask": _mask_debug_summary(fixed_mask),
        "moving_mask": _mask_debug_summary(moving_mask),
    }


def _coerce_first_float(value: Any) -> float:
    """Return the first parseable float from elastix parameter values."""
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return float(item)
            except Exception:
                continue
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _extract_elastix_final_metric_and_stop(parameter_map: Any) -> tuple[float, str]:
    """Extract best-effort final metric and stop condition from elastix outputs."""
    metric_candidates = (
        "FinalMetricValue",
        "MetricValue",
        "ExactMetricValue",
        "LastMetricValue",
    )
    metric_value = float("nan")
    metric_key = None
    for key in metric_candidates:
        metric_value = _coerce_first_float(_safe_parameter_map_get(parameter_map, key, None))
        if np.isfinite(metric_value):
            metric_key = key
            break

    stop_candidates = (
        "OptimizerStopCondition",
        "StopConditionDescription",
        "StoppingCondition",
    )
    stop_condition = "elastix"
    stop_key = None
    for key in stop_candidates:
        raw = _safe_parameter_map_get(parameter_map, key, None)
        text = str(raw[0] if isinstance(raw, (list, tuple)) and raw else raw).strip()
        if text and text.lower() != "none":
            stop_condition = text
            stop_key = key
            break

    if metric_key is not None:
        print(f"[timelapse]   extracted elastix metric from {metric_key}: {metric_value:.6g}")
    else:
        print("[timelapse]   warning: could not extract final elastix metric from result parameter map.")
    if stop_key is not None:
        print(f"[timelapse]   extracted elastix stop condition from {stop_key}: {stop_condition}")
    return metric_value, stop_condition


def _ensure_itk_elastix() -> None:
    """Helper for ensure itk elastix."""
    if itk is None:
        raise RuntimeError(
            "itk-elastix is required for this registration backend. "
            "Install with: pip install itk-elastix"
        )


def _safe_parameter_map_get(parameter_map: Any, key: str, default: Any) -> Any:
    """
    Safely read itk/elastix parameter-map keys.

    Some wrapped map-like objects raise IndexError when a missing key is
    accessed via Mapping.get(), so we avoid direct .get() here.
    """
    try:
        return parameter_map[key]
    except Exception:
        return default


def _sitk_float_to_itk(image: sitk.Image):
    """Helper for sitk float to itk."""
    dim = image.GetDimension()
    arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    itk_image = itk.image_from_array(arr)
    itk_image.SetOrigin(tuple(float(v) for v in image.GetOrigin()))
    itk_image.SetSpacing(tuple(float(v) for v in image.GetSpacing()))
    direction = np.array(image.GetDirection(), dtype=float).reshape(dim, dim)
    itk_image.SetDirection(itk.matrix_from_array(direction))
    return itk_image


def _sitk_mask_to_itk(image: sitk.Image):
    """Helper for sitk mask to itk."""
    dim = image.GetDimension()
    arr = sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)
    itk_image = itk.image_from_array(arr)
    itk_image.SetOrigin(tuple(float(v) for v in image.GetOrigin()))
    itk_image.SetSpacing(tuple(float(v) for v in image.GetSpacing()))
    direction = np.array(image.GetDirection(), dtype=float).reshape(dim, dim)
    itk_image.SetDirection(itk.matrix_from_array(direction))
    return itk_image


def _build_elastix_parameter_object(
    fixed_image: sitk.Image,
    settings: RegistrationSettings,
    use_masks: bool,
):
    """Build elastix parameter object."""
    transform_type = str(settings.transform_type).lower()
    if transform_type not in {"euler", "translation"}:
        raise ValueError(
            f"Unsupported transform_type for elastix backend: {settings.transform_type}"
        )

    optimizer = str(settings.optimizer).lower()
    if optimizer not in {"adaptive_stochastic_gradient_descent", "adaptive_stochastic"}:
        raise ValueError(
            f"Unsupported optimizer for elastix backend: {settings.optimizer}"
        )

    interpolator = str(settings.interpolator).lower()
    if interpolator not in {"linear", "nearest"}:
        raise ValueError(
            f"Unsupported interpolator for elastix backend: {settings.interpolator}"
        )

    parameter_object = itk.ParameterObject.New()
    if transform_type == "euler":
        parameter_map = parameter_object.GetDefaultParameterMap("rigid")
        parameter_map["Transform"] = ("EulerTransform",)
    else:
        parameter_map = parameter_object.GetDefaultParameterMap("translation")
        parameter_map["Transform"] = ("TranslationTransform",)

    parameter_map["FixedInternalImagePixelType"] = ("float",)
    parameter_map["MovingInternalImagePixelType"] = ("float",)

    parameter_map["Registration"] = ("MultiResolutionRegistration",)
    if interpolator == "nearest":
        parameter_map["Interpolator"] = ("NearestNeighborInterpolator",)
        parameter_map["ResampleInterpolator"] = ("FinalNearestNeighborInterpolator",)
        parameter_map["BSplineInterpolationOrder"] = ("0",)
        parameter_map["FinalBSplineInterpolationOrder"] = ("0",)
    else:
        parameter_map["Interpolator"] = ("BSplineInterpolator",)
        parameter_map["ResampleInterpolator"] = ("FinalBSplineInterpolator",)
        parameter_map["BSplineInterpolationOrder"] = ("1",)
        parameter_map["FinalBSplineInterpolationOrder"] = ("3",)
    parameter_map["Resampler"] = ("DefaultResampler",)

    parameter_map["FixedImagePyramid"] = ("FixedRecursiveImagePyramid",)
    parameter_map["MovingImagePyramid"] = ("MovingRecursiveImagePyramid",)

    parameter_map["Optimizer"] = ("AdaptiveStochasticGradientDescent",)

    if settings.metric == "correlation":
        parameter_map["Metric"] = ("AdvancedNormalizedCorrelation",)
    elif settings.metric == "mattes":
        parameter_map["Metric"] = ("AdvancedMattesMutualInformation",)
        parameter_map["NumberOfHistogramBins"] = ("32",)
        parameter_map["NumberOfFixedHistogramBins"] = ("32",)
        parameter_map["NumberOfMovingHistogramBins"] = ("32",)
        parameter_map["FixedLimitRangeRatio"] = ("0.01",)
        parameter_map["MovingLimitRangeRatio"] = ("0.01",)
        parameter_map["FixedKernelBSplineOrder"] = ("0",)
        parameter_map["MovingKernelBSplineOrder"] = ("3",)
        parameter_map["UseFastAndLowMemoryVersion"] = ("true",)
    else:
        raise ValueError(f"Unsupported elastix metric: {settings.metric}")

    parameter_map["AutomaticScalesEstimation"] = ("true",)
    parameter_map["AutomaticTransformInitialization"] = ("true",)

    if settings.initializer in ("none", "identity"):
        parameter_map["AutomaticTransformInitialization"] = ("false",)

    elif settings.initializer == "geometry":
        parameter_map["AutomaticTransformInitialization"] = ("true",)
        parameter_map["AutomaticTransformInitializationMethod"] = ("GeometricalCenter",)

    elif settings.initializer == "moments":
        parameter_map["AutomaticTransformInitialization"] = ("true",)
        parameter_map["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",)

    else:
        raise ValueError(f"Unsupported initializer: {settings.initializer}")

    if len(settings.initial_translation_voxels) != 3:
        raise ValueError(
            "initial_translation_voxels must contain exactly 3 values (x, y, z)"
        )
    initial_translation_vox = tuple(float(v) for v in settings.initial_translation_voxels)
    if any(abs(v) > 0.0 for v in initial_translation_vox):
        spacing = tuple(float(v) for v in fixed_image.GetSpacing())
        initial_translation_physical = (
            initial_translation_vox[0] * spacing[0],
            initial_translation_vox[1] * spacing[1],
            initial_translation_vox[2] * spacing[2],
        )

        if transform_type == "translation":
            parameter_map["TransformParameters"] = tuple(
                str(v) for v in initial_translation_physical
            )
        else:
            parameter_map["TransformParameters"] = (
                "0.0",
                "0.0",
                "0.0",
                str(initial_translation_physical[0]),
                str(initial_translation_physical[1]),
                str(initial_translation_physical[2]),
            )

        # Keep user-provided offset deterministic regardless of initializer.
        parameter_map["AutomaticTransformInitialization"] = ("false",)

    parameter_map["HowToCombineTransforms"] = ("Compose",)
    parameter_map["ITKTransformOutputFileNameExtension"] = ("h5",)
    parameter_map["WriteITKCompositeTransform"] = ("true",)

    parameter_map["ErodeMask"] = ("false",)
    parameter_map["ErodeFixedMask"] = ("false",)
    parameter_map["ErodeMovingMask"] = ("false",)

    # Match elastix preset style: let elastix determine default pyramid schedule
    # from NumberOfResolutions. Do NOT set ImagePyramidSchedule.
    parameter_map["NumberOfResolutions"] = (str(int(settings.number_of_resolutions)),)

    parameter_map["MaximumNumberOfIterations"] = (
        str(int(settings.number_of_iterations)),
    )

    total_voxels = int(np.prod(fixed_image.GetSize(), dtype=np.int64))
    sampling_fraction = float(settings.sampling_percentage)
    if sampling_fraction <= 0:
        raise ValueError(
            f"sampling_percentage must be > 0, got {settings.sampling_percentage}"
        )
    spatial_samples = max(1, min(total_voxels, int(round(total_voxels * sampling_fraction))))

    parameter_map["NumberOfSpatialSamples"] = (str(spatial_samples),)
    parameter_map["NewSamplesEveryIteration"] = ("true",)

    if use_masks:
        parameter_map["ImageSampler"] = ("RandomSparseMask",)
        parameter_map["MaximumNumberOfSamplingAttempts"] = ("100",)
        parameter_map["RequiredRatioOfValidSamples"] = ("0.1",)
        
    else:
        parameter_map["ImageSampler"] = ("Random",)

    parameter_map["DefaultPixelValue"] = ("0",)
    parameter_map["WriteResultImage"] = ("false",)
    parameter_map["ResultImagePixelType"] = ("short",)
    parameter_map["ResultImageFormat"] = ("mhd",)
    parameter_map["UseDirectionCosines"] = ("true",)

    # Explicit defaults to suppress common warnings
    parameter_map["ShowExactMetricValue"] = ("false",)
    parameter_map["UseMultiThreadingForMetrics"] = ("true",)
    parameter_map["UseRandomSampleRegion"] = ("false",)
    if settings.automatic_parameter_estimation:
        parameter_map["AutomaticParameterEstimation"] = ("true",)
    else:
        parameter_map["AutomaticParameterEstimation"] = ("false",)
        parameter_map["SP_A"] = (str(float(settings.sp_a)),)
        parameter_map["SigmoidInitialTime"] = ("0",)
        parameter_map["MaxBandCovSize"] = ("192",)
        parameter_map["NumberOfBandStructureSamples"] = ("10",)
        parameter_map["UseAdaptiveStepSizes"] = ("true",)
        parameter_map["UseConstantStep"] = ("false",)
        parameter_map["MaximumStepLengthRatio"] = ("1",)
        max_step_length = (
            float(settings.maximum_step_length)
            if settings.maximum_step_length is not None
            else float(min(fixed_image.GetSpacing()))
        )
        parameter_map["MaximumStepLength"] = (str(max_step_length),)
        parameter_map["NumberOfGradientMeasurements"] = (
            str(int(settings.number_of_gradient_measurements)),
        )
        parameter_map["NumberOfJacobianMeasurements"] = (
            str(int(settings.number_of_jacobian_measurements)),
        )
        parameter_map["SigmoidScaleFactor"] = (str(float(settings.sigmoid_scale_factor)),)
        parameter_map["ASGDParameterEstimationMethod"] = ("Original",)
        parameter_map["UseJacobianPreconditioning"] = ("false",)
        parameter_map["FiniteDifferenceDerivative"] = ("false",)

    parameter_object.SetParameterMap(parameter_map)
    return parameter_object


def _parameter_object_to_sitk_transform(parameter_object, dim: int) -> sitk.Transform:
    """Helper for parameter object to sitk transform."""
    parameter_map = parameter_object.GetParameterMap(0)

    transform_name = parameter_map["Transform"][0]
    params = [float(v) for v in parameter_map["TransformParameters"]]

    if dim == 3:
        center = [0.0, 0.0, 0.0]
        if "CenterOfRotationPoint" in parameter_map:
            center = [float(v) for v in parameter_map["CenterOfRotationPoint"][:3]]

        if transform_name == "TranslationTransform":
            tx = sitk.TranslationTransform(3)
            tx.SetParameters(params)
            return tx

        if transform_name != "EulerTransform":
            raise ValueError(
                f"Unsupported elastix rigid transform for 3D conversion: {transform_name}"
            )

        tx = sitk.Euler3DTransform()
        tx.SetCenter(center)
        tx.SetParameters(params)
        return tx

    if dim == 2:
        center = [0.0, 0.0]
        if "CenterOfRotationPoint" in parameter_map:
            center = [float(v) for v in parameter_map["CenterOfRotationPoint"][:2]]

        if transform_name == "TranslationTransform":
            tx = sitk.TranslationTransform(2)
            tx.SetParameters(params)
            return tx

        if transform_name != "EulerTransform":
            raise ValueError(
                f"Unsupported elastix rigid transform for 2D conversion: {transform_name}"
            )

        tx = sitk.Euler2DTransform()
        tx.SetCenter(center)
        tx.SetParameters(params)
        return tx

    raise ValueError(f"Unsupported image dimension: {dim}")


def register_images(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    settings: RegistrationSettings,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
) -> RegistrationResult:
    """Helper for register images."""
    _ensure_itk_elastix()

    if fixed_image.GetDimension() != moving_image.GetDimension():
        raise ValueError("fixed_image and moving_image must have the same dimension")

    dim = fixed_image.GetDimension()

    use_masks = settings.use_masks and fixed_mask is not None and moving_mask is not None
    total_voxels = int(np.prod(fixed_image.GetSize(), dtype=np.int64))
    sampling_fraction = float(settings.sampling_percentage)
    if sampling_fraction <= 0:
        raise ValueError(
            f"sampling_percentage must be > 0, got {settings.sampling_percentage}"
        )
    spatial_samples = max(1, min(total_voxels, int(round(total_voxels * sampling_fraction))))

    fixed_itk = _sitk_float_to_itk(sitk.Cast(fixed_image, sitk.sitkFloat32))
    moving_itk = _sitk_float_to_itk(sitk.Cast(moving_image, sitk.sitkFloat32))

    if use_masks:
        fixed_mask_itk = _sitk_mask_to_itk(sitk.Cast(fixed_mask > 0, sitk.sitkUInt8))
        moving_mask_itk = _sitk_mask_to_itk(sitk.Cast(moving_mask > 0, sitk.sitkUInt8))

    parameter_object = _build_elastix_parameter_object(
        fixed_image=fixed_image,
        settings=settings,
        use_masks=use_masks,
    )
    init_parameter_map = parameter_object.GetParameterMap(0)

    requested_initial_translation_vox = (
        float(settings.initial_translation_voxels[0]),
        float(settings.initial_translation_voxels[1]),
        float(settings.initial_translation_voxels[2]),
    )
    spacing = tuple(float(v) for v in fixed_image.GetSpacing())
    requested_initial_translation_physical = (
        requested_initial_translation_vox[0] * spacing[0],
        requested_initial_translation_vox[1] * spacing[1],
        requested_initial_translation_vox[2] * spacing[2],
    )
    init_transform_parameters = list(
        _safe_parameter_map_get(init_parameter_map, "TransformParameters", [])
    )

    if _should_trace_registration(settings):
        print(
            "[timelapse]   registration init offset "
            f"vox={list(requested_initial_translation_vox)} "
            f"physical={list(requested_initial_translation_physical)}"
        )
        print(
            "[timelapse]   elastix init TransformParameters="
            f"{init_transform_parameters}"
        )
        debug_context = _build_registration_debug_context(
            fixed_image=fixed_image,
            moving_image=moving_image,
            settings=settings,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            use_masks=use_masks,
            spatial_samples=spatial_samples,
            init_transform_parameters=init_transform_parameters,
            requested_initial_translation_vox=requested_initial_translation_vox,
            requested_initial_translation_physical=requested_initial_translation_physical,
        )
        print("[timelapse]   registration preflight context:")
        print(json.dumps(debug_context, indent=2))

    try:
        registered_itk, result_parameter_object = itk.elastix_registration_method(
            fixed_itk,
            moving_itk,
            parameter_object=parameter_object,
            fixed_mask=fixed_mask_itk,
            moving_mask=moving_mask_itk,
            log_to_console=bool(settings.debug),
        )
    except Exception as exc:
        fail_context = _build_registration_debug_context(
            fixed_image=fixed_image,
            moving_image=moving_image,
            settings=settings,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            use_masks=use_masks,
            spatial_samples=spatial_samples,
            init_transform_parameters=init_transform_parameters,
            requested_initial_translation_vox=requested_initial_translation_vox,
            requested_initial_translation_physical=requested_initial_translation_physical,
        )
        raise RuntimeError(
            "itk-elastix registration failed. Preflight context:\n"
            + json.dumps(fail_context, indent=2)
        ) from exc
    _ = registered_itk

    final_transform = _parameter_object_to_sitk_transform(
        result_parameter_object,
        dim=dim,
    )

    parameter_map = result_parameter_object.GetParameterMap(0)
    metric_value, optimizer_stop_condition = _extract_elastix_final_metric_and_stop(parameter_map)

    return RegistrationResult(
        transform=final_transform,
        metric_value=float(metric_value),
        optimizer_stop_condition=optimizer_stop_condition,
        iterations=int(settings.number_of_iterations),
        metadata={
            "backend": "itk-elastix",
            "transform_type": settings.transform_type,
            "metric": settings.metric,
            "optimizer": settings.optimizer,
            "sampling_percentage": settings.sampling_percentage,
            "automatic_parameter_estimation": settings.automatic_parameter_estimation,
            "sp_a": settings.sp_a,
            "maximum_step_length": settings.maximum_step_length,
            "sigmoid_scale_factor": settings.sigmoid_scale_factor,
            "number_of_gradient_measurements": settings.number_of_gradient_measurements,
            "number_of_jacobian_measurements": settings.number_of_jacobian_measurements,
            "number_of_resolutions": settings.number_of_resolutions,
            "interpolator": settings.interpolator,
            "initializer": settings.initializer,
            "initial_translation_voxels": [
                float(settings.initial_translation_voxels[0]),
                float(settings.initial_translation_voxels[1]),
                float(settings.initial_translation_voxels[2]),
            ],
            "initial_translation_physical": [
                float(requested_initial_translation_physical[0]),
                float(requested_initial_translation_physical[1]),
                float(requested_initial_translation_physical[2]),
            ],
            "elastix_init_transform_parameters": init_transform_parameters,
            "fixed_mask_used": use_masks,
            "moving_mask_used": use_masks,
            "elastix_transform": _safe_parameter_map_get(
                parameter_map, "Transform", ["unknown"]
            )[0],
            "elastix_transform_parameters": list(
                _safe_parameter_map_get(parameter_map, "TransformParameters", [])
            ),
            "elastix_metric_value": float(metric_value),
            "elastix_optimizer_stop_condition": optimizer_stop_condition,
        },
    )
