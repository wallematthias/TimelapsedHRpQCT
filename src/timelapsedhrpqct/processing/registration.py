from __future__ import annotations

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
    initializer: str = "geometry"
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


def _ensure_itk_elastix() -> None:
    if itk is None:
        raise RuntimeError(
            "itk-elastix is required for this registration backend. "
            "Install with: pip install itk-elastix"
        )


def _sitk_float_to_itk(image: sitk.Image):
    dim = image.GetDimension()
    arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    itk_image = itk.image_from_array(arr)
    itk_image.SetOrigin(tuple(float(v) for v in image.GetOrigin()))
    itk_image.SetSpacing(tuple(float(v) for v in image.GetSpacing()))
    direction = np.array(image.GetDirection(), dtype=float).reshape(dim, dim)
    itk_image.SetDirection(itk.matrix_from_array(direction))
    return itk_image


def _sitk_mask_to_itk(image: sitk.Image):
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
    if settings.transform_type != "euler":
        raise ValueError(
            f"Unsupported transform_type for elastix backend: {settings.transform_type}"
        )

    parameter_object = itk.ParameterObject.New()
    parameter_map = parameter_object.GetDefaultParameterMap("rigid")

    parameter_map["FixedInternalImagePixelType"] = ("float",)
    parameter_map["MovingInternalImagePixelType"] = ("float",)

    parameter_map["Registration"] = ("MultiResolutionRegistration",)
    parameter_map["Interpolator"] = ("BSplineInterpolator",)
    parameter_map["ResampleInterpolator"] = ("FinalBSplineInterpolator",)
    parameter_map["Resampler"] = ("DefaultResampler",)

    parameter_map["FixedImagePyramid"] = ("FixedRecursiveImagePyramid",)
    parameter_map["MovingImagePyramid"] = ("MovingRecursiveImagePyramid",)

    parameter_map["Optimizer"] = ("AdaptiveStochasticGradientDescent",)
    parameter_map["Transform"] = ("EulerTransform",)

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

    # Match preset exactly
    parameter_map["NumberOfSpatialSamples"] = ("2048",)
    parameter_map["NewSamplesEveryIteration"] = ("true",)

    if use_masks:
        parameter_map["ImageSampler"] = ("RandomSparseMask",)
        parameter_map["MaximumNumberOfSamplingAttempts"] = ("100",)
        parameter_map["RequiredRatioOfValidSamples"] = ("0.1",)
        
    else:
        parameter_map["ImageSampler"] = ("Random",)

    parameter_map["BSplineInterpolationOrder"] = ("1",)
    parameter_map["FinalBSplineInterpolationOrder"] = ("3",)

    parameter_map["DefaultPixelValue"] = ("0",)
    parameter_map["WriteResultImage"] = ("false",)
    parameter_map["ResultImagePixelType"] = ("short",)
    parameter_map["ResultImageFormat"] = ("mhd",)
    parameter_map["UseDirectionCosines"] = ("true",)

    # Explicit defaults to suppress common warnings
    parameter_map["ShowExactMetricValue"] = ("false",)
    parameter_map["UseMultiThreadingForMetrics"] = ("true",)
    parameter_map["UseRandomSampleRegion"] = ("false",)
    parameter_map["SP_A"] = ("20",)
    parameter_map["SigmoidInitialTime"] = ("0",)
    parameter_map["MaxBandCovSize"] = ("192",)
    parameter_map["NumberOfBandStructureSamples"] = ("10",)
    parameter_map["UseAdaptiveStepSizes"] = ("true",)
    parameter_map["UseConstantStep"] = ("false",)
    parameter_map["MaximumStepLengthRatio"] = ("1",)
    parameter_map["MaximumStepLength"] = (str(float(min(fixed_image.GetSpacing()))),)
    parameter_map["NumberOfGradientMeasurements"] = ("0",)
    parameter_map["NumberOfJacobianMeasurements"] = ("1000",)
    parameter_map["SigmoidScaleFactor"] = ("0.1",)
    parameter_map["ASGDParameterEstimationMethod"] = ("Original",)
    parameter_map["UseJacobianPreconditioning"] = ("false",)
    parameter_map["FiniteDifferenceDerivative"] = ("false",)

    parameter_object.SetParameterMap(parameter_map)
    return parameter_object


def _parameter_object_to_sitk_transform(parameter_object, dim: int) -> sitk.Transform:
    parameter_map = parameter_object.GetParameterMap(0)

    transform_name = parameter_map["Transform"][0]
    params = [float(v) for v in parameter_map["TransformParameters"]]

    if dim == 3:
        center = [0.0, 0.0, 0.0]
        if "CenterOfRotationPoint" in parameter_map:
            center = [float(v) for v in parameter_map["CenterOfRotationPoint"][:3]]

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
    _ensure_itk_elastix()

    if fixed_image.GetDimension() != moving_image.GetDimension():
        raise ValueError("fixed_image and moving_image must have the same dimension")

    dim = fixed_image.GetDimension()

    use_masks = settings.use_masks and fixed_mask is not None and moving_mask is not None
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

    registered_itk, result_parameter_object = itk.elastix_registration_method(
        fixed_itk,
        moving_itk,
        parameter_object=parameter_object,
        fixed_mask=fixed_mask_itk,
        moving_mask=moving_mask_itk,
        log_to_console=bool(settings.debug),
    )
    _ = registered_itk

    final_transform = _parameter_object_to_sitk_transform(
        result_parameter_object,
        dim=dim,
    )

    parameter_map = result_parameter_object.GetParameterMap(0)

    return RegistrationResult(
        transform=final_transform,
        metric_value=float("nan"),
        optimizer_stop_condition="elastix",
        iterations=int(settings.number_of_iterations),
        metadata={
            "backend": "itk-elastix",
            "transform_type": settings.transform_type,
            "metric": settings.metric,
            "optimizer": settings.optimizer,
            "sampling_percentage": settings.sampling_percentage,
            "number_of_resolutions": settings.number_of_resolutions,
            "interpolator": settings.interpolator,
            "initializer": settings.initializer,
            "fixed_mask_used": use_masks,
            "moving_mask_used": use_masks,
            "elastix_transform": parameter_map.get("Transform", ["unknown"])[0],
            "elastix_transform_parameters": list(
                parameter_map.get("TransformParameters", [])
            ),
        },
    )
