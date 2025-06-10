import SimpleITK as sitk
import numpy as np
import time

from . import custom_logger


def strToSitkInterpolator(interpolator: str):

    if interpolator == 'nearest':
        return sitk.sitkNearestNeighbor
    elif interpolator == 'linear':
        return sitk.sitkLinear
    elif interpolator == 'cubic':
        return sitk.sitkBSpline
    elif interpolator == 'label_gaussian':
        return sitk.sitkLabelGaussian
    elif interpolator == 'gaussian':
        return sitk.sitkGaussian
    elif interpolator == 'lanczos':
        return sitk.sitkLanczosWindowedSinc
    elif interpolator == 'bspline':
        return sitk.sitkBSpline


def equalSize(im1, im2):
    x1, y1, z1 = im1.shape
    x2, y2, z2 = im2.shape
    padded_im1 = np.zeros(np.max([im1.shape, im2.shape], axis=0))
    padded_im2 = np.zeros(np.max([im1.shape, im2.shape], axis=0))
    padded_im1[0:x1, 0:y1, 0:z1] = im1
    padded_im2[0:x2, 0:y2, 0:z2] = im2

    return padded_im1, padded_im2


class Registration:

    def __init__(self, sampling=0.01, num_of_iterations=500):
        '''
        Initialize Registration Logic class
        '''

        # initial variables
        self.baseImage = None
        self.followImage = None

        self.baseMask = None
        self.followMask = None

        self.num_of_iterations = 500

        # registration method
        self.reg = sitk.ImageRegistrationMethod()

        self.initial_transform = None

        # similarity metric
        self.reg.SetMetricAsMeanSquares()
        self.samplingStrategy = 'RANDOM'
        self.reg.SetMetricSamplingStrategy(self.reg.RANDOM)

        # change sampling percent
        self.sampling = sampling
        self.reg.SetMetricSamplingPercentage(sampling)
        self.num_of_iterations = num_of_iterations

        # interprolator
        self.interpolator = sitk.sitkBSpline
        self.interpolatorstring = 'bspline'

        # optimizer
        self.reg.SetOptimizerAsPowell()
        self.reg.SetOptimizerScalesFromPhysicalShift()

        # multi-resolution framework
        self.reg.SetShrinkFactorsPerLevel(shrinkFactors=[1, 1])
        self.reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[1.0, 0])
        self.reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.reg.AddCommand(
            sitk.sitkIterationEvent,
            lambda: self.command_iteration(
                self.reg))

        # transform
        self.FU_Transform = None

    def setRegistrationParamaters(
            self,
            baseImage: np.array,
            followImage: np.array) -> None:
        '''
        Change parameters for registration

        Args:
            baseImage (SimpleITK Image): baseline image
            followImage (SimpleITK Image): follow-up image
            sampling (float): metric sampling percentage (increase for greater accuracy at higher time cost)

        Returns:
            None
        '''
        # change image and adapt units
        self.baseShape = baseImage.shape
        self.followShape = followImage.shape

        baseImage, followImage = equalSize(
            np.asarray(baseImage), np.asarray(followImage))
        self.newShape = baseImage.shape

        self.baseImage = sitk.GetImageFromArray(np.nan_to_num(baseImage))
        self.followImage = sitk.GetImageFromArray(np.nan_to_num(followImage))

    def setRegistrationMask(
            self,
            baseMask: np.array,
            followMask: np.array) -> None:
        '''
        Change parameters for registration

        Args:
            baseMask (SimpleITK Image): baseline mask
            followMask (SimpleITK Image): follow-up mask

        Returns:
            None
        '''
        # change mask
        self.usedMask = True

        baseMask, followMask = equalSize(
            np.asarray(baseMask), np.asarray(followMask))

        self.baseMask = sitk.GetImageFromArray(
            ((np.nan_to_num(baseMask)) > 0).astype(int))
        self.followMask = sitk.GetImageFromArray(
            ((np.nan_to_num(followMask)) > 0).astype(int))

    def setInitialTransform(
            self,
            initial_rotation: np.array,
            initial_translation: np.array) -> None:
        '''
        Change the initial transform used for registration. See help message in the widget for more information on each metric.

        Args:
            initial_rotation (array): rotation angle
            initial_translation (array): translations

        Returns:
            None
        '''
        self.initial_transform = initial_rotation + initial_translation

    def setSimilarityMetric(self, metric: str) -> None:
        '''
        Change the similarity metric used for registration. See help message in the widget for more information on each metric.

        Args:
            metric (str): type of metric to use (\'mean_squares\', \'correlation\', \'mattes\', or \'ants\')

        Returns:
            None
        '''
        # determine type of metric and change
        self.metric = metric

        if metric == 'mean_squares':
            self.reg.SetMetricAsMeanSquares()
        elif metric == 'correlation':
            self.reg.SetMetricAsCorrelation()
        elif metric == 'mattes':
            self.reg.SetMetricAsMattesMutualInformation()
        elif metric == 'ants':
            self.reg.SetMetricAsANTSNeighborhoodCorrelation(2)

    def setOptimizer(self, optimizer: str) -> None:
        '''
        Change the optimizer used for registration. See help message in the widget for more information on each metric.

        Args:
            optimizer (str): type of metric to use (\'amoeba\', \'powell\', \'one_plus_one\', \'gradient\',
            \'gradient_ls\', \'gradient_reg\', or \'lbfgs2\')

        Returns:
            None
        '''
        self.optimizer = optimizer

        if optimizer == 'amoeba':
            self.reg.SetOptimizerAsAmoeba(1, self.num_of_iterations)
        elif optimizer == 'exhaustive':
            self.reg.SetOptimizerAsExhaustive(self.num_of_iterations)
        elif optimizer == 'powell':
            self.reg.SetOptimizerAsPowell()
        elif optimizer == 'one_plus_one':
            self.reg.SetOptimizerAsOnePlusOneEvolutionary()
        elif optimizer == 'gradient':
            self.reg.SetOptimizerAsGradientDescent(1, self.num_of_iterations)
        elif optimizer == 'gradient_ls':
            self.reg.SetOptimizerAsGradientDescentLineSearch(
                1, self.num_of_iterations)
        elif optimizer == 'gradient_reg':
            self.reg.SetOptimizerAsRegularStepGradientDescent(
                1, 1, self.num_of_iterations)
        elif optimizer == 'lbfgs2':
            self.reg.SetOptimizerAsLBFGS2()

    def setMultiResolution(
        self, shrinkFactors=[
            1, 1], smoothingSigmas=[
            1.0, 0]) -> None:
        '''
        Change the resolution used for registration. See help message in the widget for more information on each metric.

        Args:
            skrinkFactors (list:int): downscales the images before registration
            skrinkFactors (list:float): applies smoothing sigma before registraiton

        Returns:
            None
        '''
        self.shrinkFactors = shrinkFactors
        self.smoothingSigmas = smoothingSigmas
        self.reg.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactors)
        self.reg.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)

    def setInterpolator(self, interpolator: str) -> None:
        '''
        Change the interpolator used for registration. See help message in the widget for more information on each metric.

        Args:
            interpolator (str): type of metric to use (\'nearest\', \'linear\', \'cubic\', \'label_gaussian\', \'lanczos\', or \'lanczos\')

        Returns:
            None
        '''
        self.interpolator = strToSitkInterpolator(interpolator)
        self.interpolatorstring = interpolator

    def execute(self) -> sitk.Image:
        '''
        Run the registration algorithm

        Args:

        Returns:
            Image: registered follow up image
        '''
        self.progress = 0
        self.start = time.time()

        original_followup = self.followImage

        if self.baseMask is not None:
            #self.baseImage = sitk.Mask(self.baseImage, sitk.Cast(self.baseMask,sitk.sitkInt8), maskingValue=0, outsideValue=0)
            self.reg.SetMetricFixedMask(self.baseMask)

        if self.followMask is not None:
            #self.followImage = sitk.Mask(self.followImage, sitk.Cast(self.followMask,sitk.sitkInt8), maskingValue=0, outsideValue=0)
            self.reg.SetMetricMovingMask(self.followMask)

        initalTransform_FU_to_BL = sitk.CenteredTransformInitializer(
            self.baseImage,
            self.followImage,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)

        if self.initial_transform is not None:
            initalTransform_FU_to_BL.SetParameters(self.initial_transform)
            custom_logger.info('Applying initial transform: {}'.format(
                initalTransform_FU_to_BL.GetParameters()))

        self.reg.SetInterpolator(self.interpolator)

        self.reg.SetInitialTransform(initalTransform_FU_to_BL, inPlace=False)
        #custom_logger.info('Start registration')
        self.FU_Transform = self.reg.Execute(
            sitk.Cast(
                self.baseImage, sitk.sitkFloat64), sitk.Cast(
                self.followImage, sitk.sitkFloat64))

        custom_logger.info('Optimizer\'s stopping condition, {0}'.format(self.reg.GetOptimizerStopConditionDescription()))

        # Resample registered FU grayscale image
        #custom_logger.info('Resampling image')

        followImage_resampled = sitk.Resample(
            original_followup,
            self.baseImage,
            self.FU_Transform,
            self.interpolator,
            0.0,
            self.followImage.GetPixelID())

        self.end = time.time()

        return sitk.GetArrayFromImage(followImage_resampled)

    def transform(self, transform_arr: np.array) -> sitk.Image:
        '''
        Apply the transform to a different image

        Args:
            transform_arr
        Returns:
            Image: registered image
        '''
        transform_im = sitk.GetImageFromArray(np.asarray(transform_arr))
        resampled_im = sitk.Resample(
            transform_im,
            self.baseImage,
            self.FU_Transform,
            self.interpolator,
            0.0,
            transform_im.GetPixelID())

        return sitk.GetArrayFromImage(resampled_im)

    def command_iteration(self, method: sitk.ImageRegistrationMethod) -> None:
        '''
        custom_logger.info updates on registration status
        '''
        custom_logger.info(
            'Iteration{0:3} has a value of {1:10.5f} at position: {2:}'.format(
                method.GetOptimizerIteration(),
                method.GetMetricValue(),
                '(' +
                ', '.join(
                    ('%.9f' %
                     f) for f in method.GetOptimizerPosition()) +
                ')'))

        # update progress
        self.progress += (100 - self.progress) // 3

    def get_transform(self) -> sitk.Transform:
        '''Get registration transform'''
        return self.FU_Transform
