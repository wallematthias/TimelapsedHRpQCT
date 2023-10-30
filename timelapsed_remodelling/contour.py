import numpy as np
import time
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import ball, remove_small_objects
import itertools
from typing import Optional, Tuple
from skimage._shared.utils import check_nD
from timelapsed_remodelling.reposition import boundingbox_from_mask
from . import custom_logger

def combined_threshold(density, low_threshold=190, high_threshold=450, block_size=13):
    """
    Apply combined thresholding to the 3D density volume.
    https://doi.org/10.1016/j.bone.2021.116225

    Parameters:
        density (ndarray): The 3D density volume to be thresholded.
        low_threshold (float, optional): Threshold value for the low intensity region.
            Default is 190.
        high_threshold (float, optional): Threshold value for the high intensity region.
            Default is 450.
        block_size (int, optional): Size of the block for mean thresholding. It must be odd.
            Default is 13.

    Returns:
        ndarray: The thresholded 3D binary volume.
    """

    if block_size % 2 == 0:
        raise ValueError("The kwarg ``block_size`` must be odd! Given "
                         "``block_size`` {0} is even.".format(block_size))
    
    check_nD(density, 3)
    thresh_image = np.zeros(density.shape, 'double')

    mask = 1. / block_size * np.ones((block_size,))
    # separation of filters to speedup convolution
    ndimage.convolve1d(density, mask, axis=0, output=thresh_image, mode='reflect', cval=0)
    ndimage.convolve1d(thresh_image, mask, axis=1, output=thresh_image, mode='reflect', cval=0)
    ndimage.convolve1d(thresh_image, mask, axis=2, output=thresh_image, mode='reflect', cval=0)

    filtered_density = gaussian(density, sigma=1)

    low_mask = filtered_density > low_threshold
    local_thresh = thresh_image * low_mask

    low_image = (filtered_density * low_mask) > local_thresh
    high_image = filtered_density > high_threshold

    return remove_small_objects(high_image | low_image, min_size=64)


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    labels = ndimage.label(segmentation)[0]
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC  

def fast_binary_closing(
    image: np.ndarray,               # Input binary image as a NumPy array.
    structure: np.ndarray,           # Structuring element as a NumPy array, used for the closing operation.
    iterations: int = 1,             # Number of iterations to apply the closing operation (default is 1).
    output: np.ndarray = None,       # Optional output array to store the result (default is None).
    origin: int = -1                 # Position of the anchor within the structuring element (default is -1).
) -> np.ndarray:
    """
    Fast implementation of binary closing operation on a binary image.

    Parameters:
        image (np.ndarray): Input binary image.
        structure (np.ndarray): Structuring element used for the closing operation.
        iterations (int, optional): Number of iterations (default is 1).
        output (np.ndarray, optional): Optional output array to store the result (default is None).
        origin (int, optional): Position of the anchor within the structuring element (default is -1).

    Returns:
        np.ndarray: The result of binary closing operation as a binary NumPy array.
    """
    # Downscale the image and structuring element by a factor of 2
    downscale_factor = 0.5
    downsampled_image = ndimage.zoom(image, zoom=downscale_factor, order=0)
    downsampled_structure = ndimage.zoom(structure, zoom=downscale_factor, order=0).astype(bool)

    # Perform binary closing on the downsampled image
    closed_downsampled = ndimage.binary_closing(
        downsampled_image,
        structure=downsampled_structure,
        iterations=iterations,
        output=output,
        origin=origin
    )

    # Upscale the result back to the original image size
    upscale_factor = 1.0 / downscale_factor
    closed_upscaled = ndimage.zoom(closed_downsampled, zoom=upscale_factor, order=0).astype(bool)

    return closed_upscaled

def fast_binary_opening(
    image: np.ndarray,               # Input binary image as a NumPy array.
    structure: np.ndarray,           # Structuring element as a NumPy array, used for the closing operation.
    iterations: int = 1,             # Number of iterations to apply the closing operation (default is 1).
    output: np.ndarray = None,       # Optional output array to store the result (default is None).
    origin: int = -1                 # Position of the anchor within the structuring element (default is -1).
) -> np.ndarray:
    """
    Fast implementation of binary opening operation on a binary image.

    Parameters:
        image (np.ndarray): Input binary image.
        structure (np.ndarray): Structuring element used for the opening operation.
        iterations (int, optional): Number of iterations (default is 1).
        output (np.ndarray, optional): Optional output array to store the result (default is None).
        origin (int, optional): Position of the anchor within the structuring element (default is -1).

    Returns:
        np.ndarray: The result of binary opening operation as a binary NumPy array.
    """
    # Downscale the image and structuring element by a factor of 2
    downscale_factor = 0.5
    downsampled_image = ndimage.zoom(image, zoom=downscale_factor, order=0)
    downsampled_structure = ndimage.zoom(structure, zoom=downscale_factor, order=0).astype(bool)

    # Perform binary closing on the downsampled image
    opened_downsampled = ndimage.binary_opening(
        downsampled_image,
        structure=downsampled_structure,
        iterations=iterations,
        output=output,
        origin=origin
    )

    # Upscale the result back to the original image size
    upscale_factor = 1.0 / downscale_factor
    opened_upscaled = ndimage.zoom(opened_downsampled, zoom=upscale_factor, order=0).astype(bool)

    return opened_upscaled

def crop_pad_image(
    reference_image: np.ndarray,      # Reference image as a NumPy array.
    resize_image: np.ndarray,         # Image to be resized (cropped or padded) as a NumPy array.
    ref_img_position: Optional[Tuple[int, ...]] = None,   # Position of the reference image in the output.
    resize_img_position: Optional[Tuple[int, ...]] = None,  # Position of the resized image in the output.
    delta_position: Optional[Tuple[int, ...]] = None,  # Change in position (relative shift) between the reference and resized images.
    padding_value: int = 0   # Value used for padding when resizing the image (default is 0).
) -> np.ndarray:
    """
    Crop or pad the 'resize_image' to match the position of the 'reference_image' in the output.

    Parameters:
        reference_image (np.ndarray): Reference image as a NumPy array.
        resize_image (np.ndarray): Image to be resized (cropped or padded) as a NumPy array.
        ref_img_position (Tuple[int, ...], optional): Position of the reference image in the output (default is None).
        resize_img_position (Tuple[int, ...], optional): Position of the resized image in the output (default is None).
        delta_position (Tuple[int, ...], optional): Change in position (relative shift) between the reference and resized images.
                                                    If provided, 'ref_img_position' and 'resize_img_position' are not needed.
                                                    (default is None).
        padding_value (int, optional): Value used for padding when resizing the image (default is 0).

    Returns:
        np.ndarray: Cropped or padded image as a NumPy array with the same dtype and shape as 'resize_image'.
    """

    if (ref_img_position or resize_img_position) and delta_position:
        raise ValueError('When specifying delta position, no additional position is needed.')
    elif (not ref_img_position or not resize_img_position) and not delta_position:
        raise ValueError('Positions of both images must be specified.')

    # calculate delta_position from the two given positions
    delta_position = delta_position or np.subtract(resize_img_position, ref_img_position)

    delta_position_end = np.subtract(reference_image.shape, delta_position + resize_image.shape)

    # establishing where to pad and where to slice array
    delta_position = np.maximum(0, delta_position)
    delta_position_slice = np.abs(np.minimum(0, delta_position))

    delta_position_end = np.maximum(0, delta_position_end)
    delta_position_slice_end = np.minimum(0, delta_position_end)

    # solve problem when there was no slicing from the end. any number causes slicing (as the index is exclusive)
    delta_position_slice_end = [None if val == 0 else val for val in delta_position_slice_end]
    
    delta_position_slice_tuple = tuple(slice(x, y) for x, y in zip(delta_position_slice, delta_position_slice_end))

    # bring pad width into correct shape for np.pad function
    pad_width = np.column_stack([delta_position, delta_position_end])

    if reference_image.ndim not in (2, 3):
        raise ValueError("Function currently only supports arrays with 2 or 3 dimensions.")

    # pad, slice and ensure contiguous array memory layout
    resized_image = np.pad(resize_image, pad_width, 'constant', constant_values=padding_value)[delta_position_slice_tuple]
    return np.ascontiguousarray(resized_image)


def outer_contour(density_baseline: np.ndarray, options: dict = None, verbose: bool = False) -> np.ndarray:
    """
    Extracts the outer contour (periosteal region) from the density_baseline image.

    Parameters:
        density_baseline (np.ndarray): 3D numpy array representing the density baseline image.
        options (dict, optional): Dictionary of various parameters. Default is None.
        verbose (bool, optional): If True, prints intermediate steps. Default is False.

    Returns:
        np.ndarray: A 3D numpy array representing the outer contour (periosteal region) mask.
    """
    
    start_time = time.time()

    def print_verbose(message):
        if verbose:
            custom_logger.info("[Verbose] {}".format(message))

    if options is None:
        opt = {
            'periosteal_threshold': 300,  # 250 mg/cm**3
            'periosteal_kernelsize': 5,
            'gaussian_sigma': 1.5,
            'gaussian_truncate': 1,
            'expansion_depth': [0, 5], #0 5 
            'init_pad': 15,
            'fill_holes': True,
        }
    else:
        opt = options 

    density_baseline = density_baseline.astype('float16')  # Convert to float32 for memory efficiency

    shapeholder = np.zeros_like(density_baseline, dtype=bool)
    bb = boundingbox_from_mask(density_baseline>0)
    density_baseline = density_baseline[bb]
    
    # Thresholds
    periosteal_threshold = opt['periosteal_threshold']

    # Circular kernel size
    periosteal_kernelsize = ball(opt['periosteal_kernelsize']) 

    # Filter parameters
    gaussian_sigma = opt['gaussian_sigma']
    gaussian_truncate = opt['gaussian_truncate']

    # Sections where the density_baseline image is padded around which zeros (black) to get a mask from tightly cropped images
    init_pad_x = opt['init_pad']
    init_pad_y = opt['init_pad']
    depth = opt['expansion_depth'][0]
    density_baseline_padded = np.pad(density_baseline, ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth, depth)), mode='constant', constant_values=0)

    # Timer for padding
    pad_start_time = time.time()
    density_baseline_padded = np.pad(density_baseline, ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth, depth)), mode='constant', constant_values=0)
    pad_elapsed_time = time.time() - pad_start_time
    print_verbose("Padding Time: {:.4f} seconds".format(pad_elapsed_time))

    # Gaussian filter
    gaussian_start_time = time.time()
    density_filtered = gaussian(density_baseline_padded, sigma=gaussian_sigma, mode='mirror', truncate=gaussian_truncate)
    gaussian_elapsed_time = time.time() - gaussian_start_time
    print_verbose("Gaussian Filter Time: {:.4f} seconds".format(gaussian_elapsed_time))

    # Thresholding
    threshold_start_time = time.time()
    #density_thresholded = density_filtered > periosteal_threshold
    density_thresholded = combined_threshold(density_filtered)

    threshold_elapsed_time = time.time() - threshold_start_time
    print_verbose("Thresholding Time: {:.4f} seconds".format(threshold_elapsed_time))

    depth = opt['expansion_depth'][1]
    density_thresholded_padded = np.pad(density_thresholded, ((0, 0), (0, 0), (depth, depth)), mode='reflect')

    # Timer for getting the greatest component
    component_start_time = time.time()
    greatest_component = getLargestCC(density_thresholded_padded)
    component_elapsed_time = time.time() - component_start_time
    print_verbose("Get Greatest Component Time: {:.4f} seconds".format(component_elapsed_time))

    # Dilatation
    dilate_start_time = time.time()
    density_dilated = ndimage.morphology.binary_dilation(greatest_component, structure=periosteal_kernelsize, iterations=1)
    dilate_elapsed_time = time.time() - dilate_start_time
    print_verbose("Dilation Time: {:.4f} seconds".format(dilate_elapsed_time))

    # Determine outer region and invert
    outer_region_start_time = time.time()
    outer_region = getLargestCC(density_dilated == 0)
    outer_region = ~outer_region  # Invert the outer region mask
    outer_region_elapsed_time = time.time() - outer_region_start_time
    print_verbose("Outer Region Time: {:.4f} seconds".format(outer_region_elapsed_time))

    # Erosion to get the final mask (periosteal)
    erosion_start_time = time.time()
    mask_eroded = ndimage.morphology.binary_erosion(outer_region, structure=periosteal_kernelsize, iterations=1)
    erosion_elapsed_time = time.time() - erosion_start_time
    print_verbose("Erosion Time: {:.4f} seconds".format(erosion_elapsed_time))

    # Removal of initial padding and added slides, which were introduced against clipping of the erosion
    mask_start_time = time.time()
    mask = mask_eroded[init_pad_x:-init_pad_x, init_pad_y:-init_pad_y, depth:-depth]
    mask_elapsed_time = time.time() - mask_start_time
    print_verbose("Mask Time: {:.4f} seconds".format(mask_elapsed_time))

    # Option to fill any potential holes in the mask
    if opt['fill_holes']:
        fill_holes_start_time = time.time()
        mask = ndimage.binary_fill_holes(np.pad(mask, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=1))[:, :, 1:-1]
        fill_holes_elapsed_time = time.time() - fill_holes_start_time
    else:
        fill_holes_elapsed_time = 0.0
    
    print_verbose("Fill Holes Time: {:.4f} seconds".format(fill_holes_elapsed_time))

    elapsed_time = time.time() - start_time

    custom_logger.info("Finished Outer Contour: {:.4f} seconds".format(elapsed_time))

    shapeholder[bb] = mask
    
    return shapeholder


def inner_contour(density_baseline: np.ndarray, outer_contour: np.ndarray, site: str = 'radius', options: dict = None, verbose: bool = False) -> np.ndarray:
    """
    Extracts the inner contour (trabecular bone region) from the density_baseline image.

    Parameters:
        density_baseline (np.ndarray): 3D numpy array representing the density baseline image.
        outer_contour (np.ndarray): 3D numpy array representing the outer contour mask.
        site (str, optional): The scanned site, which can be 'radius', 'tibia', or 'misc'. Default is 'radius'.
        options (dict, optional): Dictionary of various parameters. Default is None.
        verbose (bool, optional): If True, prints intermediate steps. Default is False.

    Returns:
        np.ndarray: A 3D numpy array representing the inner contour (trabecular bone region) mask.
    """

    init_time = time.time()

    def print_verbose(message):
        if verbose:
            custom_logger.info("[Verbose] {}".format(message))

    if options is None:
        opt = {}
        opt['site'] = None
        opt['endosteal_threshold'] = 500
        opt['endosteal_kernelsize'] = 3
        opt['gaussian_sigma'] = 1.5
        opt['gaussian_truncate'] = 1
        opt['peel'] = 3
        opt['expansion_depth'] = [0, 3, 10, 3]
        opt['ipl_misc1_1_radius'] = 15
        opt['ipl_misc1_0_radius'] = 800
        opt['ipl_misc1_1_tibia'] = 25
        opt['ipl_misc1_0_tibia'] = 200000
        opt['ipl_misc1_1_misc'] = 15
        opt['ipls_misc1_0_misc'] = 800
        opt['init_pad'] = 30
    else:
        opt = options

    # Determine scanned site and resulting parameters
    if site == 'radius':
        print_verbose("Scanned site is Radius")
        ipl_misc1_1 = opt['ipl_misc1_1_radius']  # = 30
        ipl_misc1_0 = opt['ipl_misc1_0_radius']  # [voxels], NOT USED YET, represents ipl_misc1_0 in IPL_UPAT_CALGARY_EVAL_XT2_NOREG.COM
    elif site == 'tibia':
        print_verbose("Scanned site is Tibia")
        ipl_misc1_1 = opt['ipl_misc1_1_tibia']  # = 50
        ipl_misc1_0 = opt['ipl_misc1_0_tibia']  # [voxels], NOT USED YET, represents ipl_misc1_0 in IPL_UPAT_CALGARY_EVAL_XT2_NOREG.COM
    else:
        ipl_misc1_1 = opt['ipl_misc1_1_misc']  # = 30
        ipl_misc1_0 = opt['ipls_misc1_0_misc']
        print_verbose("Site is not known")

    # Parameters
    # Thresholds
    endosteal_threshold = opt['endosteal_threshold']
    # Kernel sizes
    endosteal_kernelsize = ball(opt['endosteal_kernelsize'])
    # Filter parameter
    gaussian_sigma = opt['gaussian_sigma']
    gaussian_truncate = opt['gaussian_truncate']

    shapeholder1 = np.zeros_like(density_baseline, dtype=bool)
    shapeholder2 = np.zeros_like(density_baseline, dtype=bool)

    bb = boundingbox_from_mask(density_baseline>0)
    density_baseline = density_baseline[bb]
    outer_contour = outer_contour[bb]
    
    mask = outer_contour.astype(bool)

    # Sections where the baseline_density image and mask are padded around which zeros (black) to get the mask from tightly cropped images
    padding_time = time.time()
    init_pad_x = opt['init_pad']
    init_pad_y = opt['init_pad']
    depth = opt['expansion_depth'][0]
    density_baseline_padded = np.pad(density_baseline, ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth, depth)), mode='constant', constant_values=0)  # mode='reflect'
    mask = np.pad(mask, ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth, depth)), mode='constant', constant_values=0)
    print_verbose('Padding time: {:.4f} seconds'.format(time.time() - padding_time))

    # Extracting the endosteal surface
    # Apply Gaussian filter
    gaussian_time = time.time()
    endosteal_density = np.copy(density_baseline_padded)
    endosteal_density_filtered = ndimage.gaussian_filter(endosteal_density, sigma=gaussian_sigma, order=0, mode='mirror', truncate=gaussian_truncate)  # mode='reflect'/'mirror'
    print_verbose('Gaussian time: {:.4f} seconds'.format(time.time() - gaussian_time))

    peel_time = time.time()
    peel = opt['peel']
    mask_peel = np.pad(mask, ((0, 0), (0, 0), (peel, peel)), mode='reflect')
    mask_peel = ndimage.binary_erosion(mask_peel, iterations=peel)
    mask_peel = mask_peel[:, :, peel:-peel]
    print_verbose('Peel time: {:.4f} seconds'.format(time.time() - peel_time))

    # Threshold image
    thresholding_time = time.time()
    endosteal_density_thresholded = endosteal_density_filtered > endosteal_threshold
    #endosteal_density_thresholded = combined_threshold(endosteal_density_filtered)

    endosteal_density_thresholded = np.logical_and(endosteal_density_thresholded, mask_peel)

    endosteal_masked_time = time.time()
    density_baseline_cropped = density_baseline_padded[boundingbox_from_mask(endosteal_density_thresholded)]
    print_verbose('Thresholding time: {:.4f} seconds'.format(time.time() - thresholding_time))

    # Mask cortical bone away. Results in the trabecular bone region only (endosteal)
    endosteal_masked_time = time.time()
    endosteal_masked = np.logical_and(np.invert(endosteal_density_thresholded), mask_peel)
    print_verbose('Endosteal masked time: {:.4f} seconds'.format(time.time() - endosteal_masked_time))

    # Expanding section (against clipping)
    endosteal_padded_time = time.time()
    depth = opt['expansion_depth'][1]
    endosteal_padded = np.pad(endosteal_masked, ((0, 0), (0, 0), (depth, depth)), mode='reflect')
    print_verbose('Endosteal padded time: {:.4f} seconds'.format(time.time() - endosteal_padded_time))

    # Extract the greatest component
    endosteal_component_time = time.time()
    endosteal_component = getLargestCC(endosteal_padded)
    print_verbose('Endosteal component time: {:.4f} seconds'.format(time.time() - endosteal_component_time))

    # 1st Erosion-CL-Dilation loop
    endosteal_eroded_time = time.time()
    endosteal_eroded = ndimage.binary_erosion(endosteal_component, structure=endosteal_kernelsize, iterations=1)
    print_verbose('Endosteal eroded time: {:.4f} seconds'.format(time.time() - endosteal_eroded_time))

    # Dilation
    endosteal_dilated_time = time.time()
    endosteal_dilated = ndimage.binary_dilation(endosteal_eroded, structure=endosteal_kernelsize, iterations=1)
    print_verbose('Endosteal dilated time: {:.4f} seconds'.format(time.time() - endosteal_dilated_time))

    # Removal of added slides that were introduced against the clipping of the erosion
    endosteal_dilated_time = time.time()
    endosteal_dilated = endosteal_dilated[:, :, depth:-depth]
    print_verbose('Endosteal dilated time: {:.4f} seconds'.format(time.time() - endosteal_dilated_time))

    # Crop the image
    endosteal_cropped_time = time.time()
    endosteal_cropped = endosteal_dilated[boundingbox_from_mask(endosteal_density_thresholded)]
    print_verbose('Endosteal cropped time: {:.4f} seconds'.format(time.time() - endosteal_cropped_time))

    # Expanding section (against clipping)
    endosteal_cropped_padded_time = time.time()
    bound_x = opt['init_pad']
    bound_y = opt['init_pad']
    depth = opt['expansion_depth'][2]
    endosteal_cropped_padded = np.pad(endosteal_cropped, ((bound_x, bound_x), (bound_y, bound_y), (0, 0)), mode='constant', constant_values=0)
    endosteal_cropped_padded = np.pad(endosteal_cropped_padded, ((0, 0), (0, 0), (depth, depth)), mode='reflect')
    print_verbose('Endosteal cropped padded time: {:.4f} seconds'.format(time.time() - endosteal_cropped_padded_time))

    # !! Large close/open sequence to smooth the contour
    endosteal_closed_time = time.time()
    endosteal_closed = fast_binary_closing(endosteal_cropped_padded, structure=ball(10), iterations=1)
    print_verbose('Endosteal closed time: {:.4f} seconds'.format(time.time() - endosteal_closed_time))

    # Put corners back, which may have been deleted by the opening process
    endosteal_opened_time = time.time()
    endosteal_opened = fast_binary_opening(endosteal_closed, structure=ball(10))
    print_verbose('Endosteal opened time: {:.4f} seconds'.format(time.time() - endosteal_opened_time))

    # Removal of added slides and x- and y- distance, which were introduced against clipping of the erosion
    endosteal_closed_time = time.time()
    endosteal_closed = endosteal_closed[bound_x:-bound_x, bound_y:-bound_y, depth:-depth]
    endosteal_opened = endosteal_opened[bound_x:-bound_x, bound_y:-bound_y, depth:-depth]
    print_verbose('Endosteal closed time: {:.4f} seconds'.format(time.time() - endosteal_closed_time))

    # 2nd Erosion-CL-Dilation loop
    corners_time = time.time()
    corners = np.subtract(endosteal_closed.astype(int), endosteal_opened.astype(int)).astype(bool)
    print_verbose('Corners time: {:.4f} seconds'.format(time.time() - corners_time))

    expansion_depth_time = time.time()
    depth = opt['expansion_depth'][3]
    corners_padded = np.pad(corners, ((0, 0), (0, 0), (depth, depth)), mode='reflect')
    print_verbose('Expansion depth time: {:.4f} seconds'.format(time.time() - expansion_depth_time))

    corners_time = time.time()
    corn_ero = ndimage.binary_erosion(corners_padded, structure=ball(3), iterations=1)
    corn_cl = ndimage.binary_dilation(corn_ero, structure=ball(3), iterations=1)
    corners = corn_cl[:, :, depth:-depth]

    trab_mask = np.add(corners, endosteal_opened)

    bound_x = opt['init_pad']
    bound_y = opt['init_pad']
    depth = ipl_misc1_1
    trab_mask_padded = np.pad(trab_mask, ((bound_x, bound_x), (bound_y, bound_y), (0, 0)), mode='constant', constant_values=0)
    trab_mask_padded = np.pad(trab_mask_padded, ((0, 0), (0, 0), (depth, depth)), mode='reflect')
    print_verbose('Corners erosion time: {:.4f} seconds'.format(time.time() - corners_time))

    # Final closing
    final_closing_time = time.time()
    trab_close = fast_binary_closing(trab_mask_padded, structure=ball(ipl_misc1_1), iterations=1)

    trab_mask = trab_close[bound_x:-bound_x, bound_y:-bound_y, depth:-depth]
    print_verbose('Final closing time: {:.4f} seconds'.format(time.time() - final_closing_time))

    # Resize the masks to the initial image (density_baseline)
    image_bounds_time = time.time()
    image_bounds = boundingbox_from_mask(endosteal_density_thresholded, 'list')
    empty_image = np.zeros(density_baseline.shape)
    print_verbose('Image bounds time: {:.4f} seconds'.format(time.time() - image_bounds_time))

    resized_trab_mask_time = time.time()
    resized_trab_mask = crop_pad_image(empty_image, trab_mask, [0, 0, 0], [image_bounds[0][0] - init_pad_x, image_bounds[1][0] - init_pad_y, 0], padding_value=0)
    print_verbose('Resized trab mask time: {:.4f} seconds'.format(time.time() - resized_trab_mask_time))

    resized_cort_mask = outer_contour.astype(bool)
    resized_cort_mask[resized_trab_mask.astype(bool)] = False

    shapeholder1[bb] = resized_trab_mask
    shapeholder2[bb] = resized_cort_mask
    custom_logger.info('Finished Inner Contour: {:.4f} seconds'.format(time.time() - init_time))

    return shapeholder1, shapeholder2
