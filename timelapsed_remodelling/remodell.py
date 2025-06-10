import time 
import numpy as np
from skimage.morphology import remove_small_objects
from .contour import combined_threshold
from . import custom_logger
from skimage.filters import gaussian

def hrpqct_remodelling_logic(baseline, followup, mask=None, threshold=225, cluster=12, segmask=None):
    """
    Apply remodelling logic to analyze changes between two images and classify the regions.
    
    Parameters:
        baseline (numpy.ndarray): The baseline image.
        followup (numpy.ndarray): The follow-up image.
        mask (numpy.ndarray, optional): A boolean array indicating the region of interest. If not provided,
            the whole image will be considered as the region of interest.
        threshold (int, optional): The grayscale difference threshold to detect significant changes in pixel values.
            Pixels with a difference greater than this threshold are considered as forming new bone (formation),
            and pixels with a difference smaller than the negative threshold are considered as bone resorption.
            Defaults to 225.
        cluster (int, optional): The minimum size (in pixels) of connected regions to be considered as bone formation
            or resorption. Smaller regions will be ignored. 
            For XT1: Defaults to 5.
            For XT2: Defaults to 12.

    Returns:
        numpy.ndarray: A labeled array representing the classified regions:
            0 - Background
            1 - Resorption (regions where bone has been resorbed)
            2 - Quiescence (regions with no significant changes between baseline and follow-up)
            3 - Formation (regions where new bone has formed)
    """

    rem_time = time.time()
    
    # If mask is not provided, create a boolean array with the same shape as the baseline image, indicating the whole image as the region of interest.
    if mask is None:
        mask = np.ones_like(baseline, dtype=bool)
    else:
        # Ensure the mask is a boolean array.
        mask = np.asarray(mask, dtype=bool)
        
    # Apply a thresholding function (combined_threshold) to segment the baseline and follow-up images based on pixel intensity.
    
    if segmask is not None: 
        # Segment Baseline according to XT2 segmentation
        baseline = gaussian(baseline, sigma=1.2)
        followup = gaussian(followup, sigma=1.2)
        seg_baseline = np.zeros_like(baseline).astype(bool)
        seg_baseline[(baseline>320) & (segmask['b_trab']>0)]=True
        seg_baseline[(baseline>450) & (segmask['b_cort']>0)]=True
        
        # Segment Followup according to XT2 segmentation
        seg_followup = np.zeros_like(followup).astype(bool)
        seg_followup[(followup>320) & (segmask['f_trab']>0)]=True
        seg_followup[(followup>450) & (segmask['f_cort']>0)]=True
        
    else:
        seg_baseline = combined_threshold(baseline)
        seg_followup = combined_threshold(followup)
        baseline = gaussian(baseline, sigma=1.2)
        followup = gaussian(followup, sigma=1.2)
        
    seg_baseline *= mask
    seg_followup *= mask
    
    # Calculate the grayscale difference between the follow-up and baseline images, considering only the region of interest.
    grayscale_difference = (followup-baseline) * mask

    # Classify regions of bone formation and resorption based on thresholding the segmented images and grayscale difference.
    binary_formation = ~seg_baseline & seg_followup
    binary_resorption = seg_baseline & ~seg_followup
    gray_formation = grayscale_difference > threshold 
    gray_resorption = grayscale_difference < -threshold 

    # Remove small regions from bone formation and resorption using morphological operations.
    formation = remove_small_objects(binary_formation & gray_formation, min_size=cluster)
    resorption = remove_small_objects(binary_resorption & gray_resorption, min_size=cluster)

    # Classify regions of quiescence where no significant changes are observed.
    quiescence = seg_baseline & ~formation & ~resorption

    # Create a labeled array to represent different types of regions.
    remodelling_image = np.zeros_like(seg_baseline, dtype=int)
    remodelling_image[resorption] = 1
    remodelling_image[quiescence] = 2
    remodelling_image[formation] = 3

    formation_fraction = np.sum(remodelling_image==3)/np.sum(remodelling_image==2)
    resorption_fraction = np.sum(remodelling_image==1)/np.sum(remodelling_image==2)

    custom_logger.info('Finished Remodelling time: {:.4f} seconds'.format(time.time() - rem_time))
    custom_logger.info('FV/BV: {:.4f}'.format(formation_fraction))
    custom_logger.info('RV/BV: {:.4f}'.format(resorption_fraction))
          
    return remodelling_image