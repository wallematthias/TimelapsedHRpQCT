import os
from glob import glob
import h5py
import copy
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from typing import List 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import uuid
from aim import aim
from motionscore.cli import automatic_motion_score

from .contour import outer_contour, inner_contour, combined_threshold, getLargestCC
from .transform import TimelapsedTransformation
from .register import Registration
from .remodell import hrpqct_remodelling_logic
from .reposition import (
    pad_array_centered, pad_and_crop_position, boundingbox_from_mask, update_pos_with_bb
)
from aim.resize_reposition_image import crop_pad_image
from .visualise import dict_to_vtkFile

from . import custom_logger

import warnings
from functools import reduce

# Suppress all warnings (not recommended for production code)
#warnings.filterwarnings("ignore")

def write_mha_file(data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None, file_path="output.mha"):
    """
    Write a 3D NumPy array to an MHA file using SimpleITK.

    Parameters:
        data (numpy.ndarray): The 3D NumPy array.
        spacing (tuple): Voxel spacing (default is (1.0, 1.0, 1.0)).
        origin (tuple): Image origin (default is (0.0, 0.0, 0.0)).
        direction (tuple): Image direction as a 3x3 matrix (default is None).
        file_path (str): Output MHA file path (default is "output.mha").
    """
    # Create a SimpleITK image from the NumPy array

    data = np.swapaxes(data, 0, 2)
    spacing = spacing[::-1]
    origin = origin[::-1]

    image = sitk.GetImageFromArray(data)

    # Ensure the image is of type float32
    image = sitk.Cast(image, sitk.sitkUInt16)
    
    # Set image metadata
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)

    # Write the image to the MHA file
    sitk.WriteImage(image, file_path)

    print(f"Image written to {file_path}")

def plot_remodelling(array, image_key, path=None):
    # Get the shape of the input array
    z_dim, x_dim, y_dim = array.shape
    
    # Find the central slice along the z-axis
    central_slice = array[z_dim // 2, :, :]
    
    # Define color mapping for 0, 1, 2, and 3
    cmap = plt.cm.colors.ListedColormap(['white', 'purple', 'gray', 'orange'])
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Display the central slice with colors based on the colormap
    ax.imshow(central_slice, cmap=cmap, interpolation='nearest')
    
    # Show the colorbar for reference
    cbar = plt.colorbar(ax.imshow(central_slice, cmap=cmap, interpolation='nearest'), ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', 'Resorption', 'Quiescence', 'Formation'])
    
    ax.axis('off')

    # Set plot title
    file_name = f'{image_key}_remodelling.png'
    plt.title(image_key)  # Set the title of the plot
    plt.savefig(os.path.join(path,file_name))

def plot_slice_with_masks_and_save(image, masks, image_key, path=None):
    """
    Plot one z-slice of the 3D grayscale image with masks overlaid, save the plot,
    and use the image_key as the title.
    
    Args:
        image (ndarray): 3D grayscale image.
        masks (list of ndarray): List of 3D binary masks.
        image_key (str): Title for the plot and part of the file name.
    """
    # Select a z-slice
    z_slice = image.shape[2] // 2  # You can modify this to select a specific z-slice
    
    # Create a colormap with transparent colors
    cmap = ListedColormap(['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'cyan', 'magenta'])
    
    # Plot the grayscale image
    plt.imshow(image[:, :, z_slice], cmap='gray')
    
    # Overlay masks with different transparent colors
    for idx, mask in enumerate(masks):
        mask_slice = mask[:, :, z_slice]
        mask_rgba = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 4))
        mask_rgba[mask_slice > 0] = [idx/len(masks), 1, 1, 0.5]  # Set alpha channel to 0.5 for transparency
        plt.imshow(mask_rgba, cmap=cmap, interpolation='none', aspect='auto')
    
    # Generate a random string for the file name
    random_string = str(uuid.uuid4())
    
    # Save the plot with the randomly generated string and image_key in the file name
    file_name = f'{image_key}_masks.png'
    plt.title(image_key)  # Set the title of the plot
    plt.savefig(os.path.join(path,file_name))
    
    # Show the plot (optional)
    plt.show()


def save_dict_to_hdf5(file_name: str, dictionary: dict):
    """
    Save a dictionary to an HDF5 file.

    Parameters:
        file_name (str): The name of the HDF5 file to create or overwrite.
        dictionary (dict): The dictionary to be saved as an HDF5 dataset.
    """
    with h5py.File(file_name, "w") as h5file:
        for key, value in dictionary.items():
            h5file.create_dataset(str(key), data=value)

def save_numpy_array_as_mha(numpy_array: np.ndarray, file_path: str):
    """
    Save a NumPy array as an MHA (MetaImage) file using SimpleITK.

    Parameters:
        numpy_array (np.ndarray): The NumPy array to be saved as an MHA file.
        file_path (str): The file path for saving the MHA file.
    """
    # Convert the numpy array to a SimpleITK image
    image = sitk.GetImageFromArray(numpy_array.astype(int))

    # Save the image as an MHA file
    sitk.WriteImage(image, file_path)

def load_numpy_array_from_mha(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from an MHA (MetaImage) file using SimpleITK.

    Parameters:
        file_path (str): The file path to the MHA file to be loaded.

    Returns:
        np.ndarray: The NumPy array loaded from the MHA file.
    """
    # Read the MHA file as a SimpleITK image
    image = sitk.ReadImage(file_path)

    # Convert the SimpleITK image to a NumPy array
    numpy_array = sitk.GetArrayFromImage(image)

    return numpy_array


class TimelapsedImageSeries:
    """
    Initialize a TimelapsedImageSeries object.

    Parameters:
        site (str): The site associated with the image series.
        name (str): The name of the image series.
        verbose (bool): If True, enable verbose mode for debugging (default: False).
        resolution (float or None): The desired resolution of the images (default: None).
        crop (bool): If True, crop images to the smallest bounding box (default: False).
    """

    
    def __init__(self, site, name, verbose=False, resolution=None, crop=False):
        self.data = {}
        self.reg_data = {}
        self.path_data = {}
        self.motion_data = {}
        self.registration = None
        self.transform = TimelapsedTransformation()
        self.position = {}
        self.voxelsize= None
        self.image_contour_mapping = {}
        self.contour_identifier = 'contour'
        self.image_identifier = 'image'
        self.shape = (0,0,0)
        self.site = site
        self.verbose = verbose
        self.name = name.split('.AIM')[0]
        self.cluster = None
        self.threshold = None
        self.analysis_results = []
        self.resolution = resolution
        self.crop = crop

        # Create the CustomLogger instance and activate it
        custom_logger.set_log_file(self.name)

    
    def add_image(self, image_name, image_path):
        """
        Add an image to the data dictionary.
    
        Parameters:
        - image_name: str, the unique name/key for the image.
        - image_path: str, the path to the image file.
    
        Raises:
        - ValueError: If an image with the same image_name already exists.
    
        """
        
        # Check if the image_id already exists in the dictionary
        if self.get_image(image_name) in self.data:
            raise ValueError(f"Image with ID '{image_name}' already exists.")
        else:
            # Load the image
            image_data, position = self.load_aim(image_path,crop=self.crop)

            # Add the image and position to the series
            self.data[self.get_image(image_name)] = image_data
            self.position[self.get_image(image_name)] = position
            self.path_data[self.get_image(image_name)] = image_path
            
            # get the new shape of the series and update size
            self.shape = np.amax([self.shape, image_data.shape],axis=0)
    
    def add_contour_to_image(self, image_name, contour_name, contour_path):
        """
        Add a contour to an existing image in the data dictionary and associate it with the image.
    
        Parameters:
            image_name (str): The name/key of the image to which the contour belongs.
            contour_name (str): The unique name/key for the contour.
            contour_path (str): The path to the contour file.
    
        Raises:
            ValueError: If the specified image_name does not exist in the data dictionary.
        """
        if self.get_image(image_name) in self.data:
            contour_data, position = self.load_aim(contour_path, crop=self.crop)
    
            # Get the position from the image
            new_position = self.position[self.get_image(image_name)]
            ref_image = self.data[self.get_image(image_name)]
            new_shape = ref_image.shape
            #contour_data = pad_and_crop_position(contour_data, position, new_position,new_shape)
            contour_data = crop_pad_image(ref_image,contour_data, ref_img_position=new_position,
                   resize_img_position=position)
        
            # Add the cropped contour to the series
            self.data[self.get_contour(contour_name, image_name)] = contour_data
            self.position[self.get_contour(contour_name, image_name)] = new_position
    
            # Get the new shape of the series and update all to be the same
            self.shape = np.amax([self.shape, contour_data.shape], axis=0)

            if self.get_image(image_name) not in self.image_contour_mapping:
                self.image_contour_mapping[self.get_image(image_name)] = []
            self.image_contour_mapping[self.get_image(image_name)].append(self.get_contour(contour_name, image_name))
        else:
            raise ValueError(f"Image with name '{image_name}' not found.")


    def generate_contour(self, image_name, path=None):
        """
        Generate 3 contours (OUT_MASK, TRAB_MASK, and CORT_MASK) for an existing image in the data dictionary 
        and associate them with the image.
    
        Parameters:
            image_name (str): The name/key of the image to which the contours belong.
            path (str, optional): The path where the generated contours will be saved as MHA files. 
                                  If not provided, the contours won't be saved.
        """
        key = self.get_image(image_name)
    
        if path is not None:
            path = os.path.join(path, self.name)
            if not os.path.exists(path):
                os.makedirs(path)
    
        # Generate or load outer contour
        if path is not None:
            name_str = os.path.basename(self.path_data[key]).split('.AIM')[0]
            OUT_MASK_PATH = os.path.join(path, name_str + '_OUT_MASK.mha')
        
        if (path is not None) and os.path.exists(OUT_MASK_PATH):
            outer_mask = load_numpy_array_from_mha(OUT_MASK_PATH)
            custom_logger.info('Loaded: {}'.format(OUT_MASK_PATH))
        else:
            outer_mask = outer_contour(self.data[key], verbose=self.verbose)
    
        # Add outer contour to the series
        self.data[self.get_contour('OUT_MASK', image_name)] = outer_mask
        self.position[self.get_contour('OUT_MASK', image_name)] = self.position[self.get_image(image_name)]
    
        # Generate or load trab and cort masks
        if path is not None:
            TRAB_MASK_PATH = os.path.join(path, name_str + '_TRAB_MASK.mha')
            CORT_MASK_PATH = os.path.join(path, name_str + '_CORT_MASK.mha')
    
        if (path is not None) and os.path.exists(TRAB_MASK_PATH) and os.path.exists(CORT_MASK_PATH):
            trab_mask = load_numpy_array_from_mha(TRAB_MASK_PATH)
            cort_mask = load_numpy_array_from_mha(CORT_MASK_PATH)
        else:
            trab_mask, cort_mask = inner_contour(self.data[key], outer_mask, site=self.site, verbose=self.verbose)
    
        # Add trab and cort masks to the series
        self.data[self.get_contour('TRAB_MASK', image_name)] = trab_mask
        self.position[self.get_contour('TRAB_MASK', image_name)] = self.position[key]
        self.data[self.get_contour('CORT_MASK', image_name)] = cort_mask
        self.position[self.get_contour('CORT_MASK', image_name)] = self.position[key]
    
        # Create mapping between Contour and Images
        if self.get_image(image_name) not in self.image_contour_mapping:
            self.image_contour_mapping[key] = []
    
        # Map the masks with the images (this can be changed in the future)
        self.image_contour_mapping[key].append(self.get_contour('OUT_MASK', image_name))
        self.image_contour_mapping[key].append(self.get_contour('TRAB_MASK', image_name))
        self.image_contour_mapping[key].append(self.get_contour('CORT_MASK', image_name))
    
        # Save masks as MHA files if path is provided
        if path is not None:
            save_numpy_array_as_mha(outer_mask, OUT_MASK_PATH)
            save_numpy_array_as_mha(trab_mask, TRAB_MASK_PATH)
            save_numpy_array_as_mha(cort_mask, CORT_MASK_PATH)

    
    def register(self, registration, mask_nr=0, path=None, registration_mode = "sequential"):
        """
        Register images in the timelapse using the provided registration method.

        Parameters:
            registration: The registration object to use for image registration.
            mask_nr (int): The index of the contour mask to use for registration (default: 0).
        """

        if path is not None:
            TRANSFORM_PATH = os.path.join(path,self.name)
        else:
            TRANSFORM_PATH = ''
        
        self.update_size_and_position()
        self.registration = registration

        imkeys = self.get_all_images()
        
        # Loop through the image keys
        for followup_idx in range(1, len(imkeys)):
            if registration_mode == "baseline":
                # Register to the baseline image (index 0)
                baseline_num = imkeys[0]
                followup_num = imkeys[followup_idx]
            else:
                # Register consecutively
                baseline_num = imkeys[followup_idx - 1]
                followup_num = imkeys[followup_idx]        

        #for baseline_num, followup_num in zip(imkeys[:-1],imkeys[1:]):

            baseline_key = self.get_image(baseline_num)
            followup_key = self.get_image(followup_num)
            
            custom_logger.info('Registering: {} to {}'.format(baseline_key, followup_key))
            registration.setRegistrationParamaters(self.data[baseline_key], self.data[followup_key])

            # Check if registration masks were set
            if len(self.get_all_contours())>=len(self.get_all_images()):
                
                registration.setRegistrationMask(
                    #self.data[self.get_contours_from_image(baseline_key)[mask_nr]],
                    #self.data[self.get_contours_from_image(followup_key)[mask_nr]])
                    np.sum([self.data[key] for key in self.get_contours_from_image(baseline_key)],axis=0)>0,
                    np.sum([self.data[key] for key in self.get_contours_from_image(followup_key)],axis=0)>0)

                
                
            # Here we start the registration
            tpath = os.path.join(TRANSFORM_PATH,'{}_{}_{}.tfm').format(baseline_num,followup_num,self.name)

            if os.path.exists(tpath):
                self.transform.load_transform(tpath)
                custom_logger.info('Loaded Transform: {}'.format(tpath))
            else:
                registration.execute()

                # Get the transform and add it to the transform class that handles automatic
                # compounding of transforms
                transform = registration.get_transform()
                metric = registration.reg.GetMetricValue()
                self.transform.add_transform(transform, baseline_num, followup_num, metric)

                if path is not None: 
                    self.transform.save_transform(TRANSFORM_PATH)

    def motion_grade(self,outpath=None):

        path = os.path.join(outpath, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
            
        for im in self.get_all_images():

            key = self.get_image(im)
            custom_logger.info(os.path.basename(self.path_data[key]))
            name = os.path.basename(self.path_data[key]).split('.')[0]
            mscore, mscorevalue = automatic_motion_score(
                self.data[key], outpath=os.path.join(path,name), stackheight=168)

            custom_logger.info('Motion Score {}: {}'.format(key,mscore))
            
            self.motion_data[im] = mscore

    def debug(self,outpath=None):
        
        path = os.path.join(outpath, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.update_size_and_position()
        for im in self.get_all_images():
            image_key = self.get_image(im)
            image = self.data[image_key]
            contours = [self.data[key] for key in self.get_contours_from_image(image_key)]
            plot_slice_with_masks_and_save(image, contours, image_key, path)
            
    def transform_all(self,transform_to):
        """
        Transform all images and contours to a common reference space.

        Parameters:
            transform_to (str): The name/key of the image to which all data will be transformed.
        """
        # This is how easy it is to transform all masks/images into a certain space! 
        nodes = list(self.transform.data.nodes)
        #transform_to = self.get_image(transform_to)
        
        for data_key in self.get_all_data():
            transform_from = [key for key in nodes if key in data_key][0]
            #custom_logger.info(transform_from)
            # Transform if necessary
            if transform_to != transform_from:
                 transformed_image = self.transform.transform(
                    self.data[data_key], transform_to, transform_from)
            # Otherwise don't
            else:
                transformed_image = self.data[data_key]

            # Save registered data
            self.reg_data[data_key] = transformed_image
    
    def analyse(self, baseline, followup, threshold=225, cluster=12, outpath=None):
        """
        Analyze the timelapse series and calculate various metrics.
    
        Parameters:
            baseline (str): The name/key of the baseline image.
            followup (str): The name/key of the followup image.
            threshold (int): The threshold value for analysis (default: 225).
            cluster (int): The cluster value for analysis (default: 5).
        """
        self.threshold = threshold
        self.cluster = cluster
    
        # Get common region between baseline and followup images 
        # This also transforms all images to "baseline"
        common_region = self.common_region(baseline)
    
        # Load baseline and followup image data (registered to baseline)
        baseline_data = self.reg_data[self.get_image(baseline)]
        followup_data = self.reg_data[self.get_image(followup)]

        # This part is a bit hacky and only for the silly trab/cort segmentation
        baseline_masks = self.get_contours_from_image(baseline)
        followup_masks = self.get_contours_from_image(followup)

        # Find the first string containing "Trab" (case-insensitive)
        btrab_key = next((s for s in baseline_masks if 'trab' in s.lower()), None)
        bcort_key = next((s for s in baseline_masks if 'cort' in s.lower()), None)
        ftrab_key = next((s for s in followup_masks if 'trab' in s.lower()), None)
        fcort_key = next((s for s in followup_masks if 'cort' in s.lower()), None)

        # Get the according masks
        # Assuming self.reg_data is a dictionary
        segmask = {
            'b_trab': copy.deepcopy(self.reg_data.get(btrab_key, {})),
            'b_cort': copy.deepcopy(self.reg_data.get(bcort_key, {})),
            'f_trab': copy.deepcopy(self.reg_data.get(ftrab_key, {})),
            'f_cort': copy.deepcopy(self.reg_data.get(fcort_key, {}))
        }

        # Perform HR-pQCT remodelling analysis
        remodelling_image = hrpqct_remodelling_logic(baseline_data, followup_data, mask=common_region, segmask=segmask, threshold=threshold, cluster=cluster)
        
        plot_remodelling(remodelling_image, f'{baseline}_{followup}_remodelling', os.path.join(outpath,self.name))
        common_position = self.position[self.get_image(baseline)]
        self.common_position = common_position
        write_mha_file(
            remodelling_image,
            spacing=[self.voxelsize,]*3, 
            #origin = [int(x) for x in common_position],
            file_path=os.path.join(outpath,self.name,f'{self.name}_remodelling_{baseline}_{followup}.mha')
            )
    
        # Calculate metrics and save results
        df = self.calculate_metrics_and_save_results(baseline, followup, remodelling_image,outpath=outpath)
    
        self.analysis_results.append(df)
        custom_logger.info(df)


            
            
    def calculate_metrics_and_save_results(self, baseline, followup, remodelling_image, outpath=None):
        """
        Calculate metrics based on HR-pQCT remodelling image and save the results to a DataFrame.
    
        Parameters:
            baseline (str): The name/key of the baseline image.
            followup (str): The name/key of the followup image.
            remodelling_image (numpy array): HR-pQCT remodelling image.
    
        Returns:
            pd.DataFrame: DataFrame containing the calculated metrics.
        """
        cols = ['IM','BASE_NAME','FOLLOW_NAME' ,'SITE','BASE_FOLL','BASE_MOTION','FOLLOW_MOTION', 'THR', 'CLUSTER', 'ROI', 'MEAS', 'VAL']
        baseline_name = os.path.basename(self.path_data[self.get_image(baseline)]).split('.')[0]
        followup_name = os.path.basename(self.path_data[self.get_image(followup)]).split('.')[0]
        
        baseline_masks = ['FULLMASK',] + self.get_contours_from_image(baseline)
        followup_masks = ['FULLMASK',] + self.get_contours_from_image(followup)
        
        
        dfs = []
        for bmask_key, fmask_key in zip(baseline_masks, followup_masks):
            
            # We are adding the full region for further analysis
            if bmask_key == 'FULLMASK':
                bmask = np.ones_like(remodelling_image)>0
                fmask = np.ones_like(remodelling_image)>0
                common_str = bmask_key
            else:
                bmask = self.reg_data[bmask_key]
                fmask = self.reg_data[fmask_key]
                common_str = bmask_key.split(self.contour_identifier)[1].split(self.image_identifier)[0].replace('_', '')
    
            mask = (bmask > 0) #| (fmask > 0) #just based on baseline is better
            FVBV = np.sum(remodelling_image[mask] == 3) / np.sum(remodelling_image[mask] == 2)
            RVBV = np.sum(remodelling_image[mask] == 1) / np.sum(remodelling_image[mask] == 2)
            BV = np.sum(remodelling_image[mask] == 2)

            try:
                bmotion = self.motion_data[baseline]
                fmotion = self.motion_data[followup]
            except:
                bmotion = np.nan
                fmotion = np.nan

            dfs.append(pd.DataFrame(
                [[self.name, baseline_name, followup_name, self.site, '{}_{}'.format(baseline, followup), bmotion, fmotion, self.threshold, self.cluster, common_str,
                  'FVBV', FVBV]], columns=cols))
            dfs.append(pd.DataFrame(
                [[self.name, baseline_name, followup_name, self.site, '{}_{}'.format(baseline, followup), bmotion, fmotion, self.threshold, self.cluster, common_str,
                  'RVBV', RVBV]], columns=cols))
            dfs.append(pd.DataFrame(
                [[self.name, baseline_name, followup_name, self.site, '{}_{}'.format(baseline, followup), bmotion, fmotion, self.threshold, self.cluster, common_str,
                  'BV', BV]], columns=cols))
    
            write_mha_file(
                mask.astype(int),
                spacing=[self.voxelsize,]*3, 
                #origin = [int(x) for x in self.common_position],
                file_path=os.path.join(outpath,self.name,f'{self.name}_{common_str}_{baseline}_{followup}.mha')
                )

        df = pd.concat(dfs)
        return df        

    def common_region(self,image):
        """
        Calculate the common region among all masks and images.

        Parameters:
            image (str): The name/key of the image used as a reference for common region calculation.

        Returns:
            np.ndarray: The combined binary mask representing the common region.
        """
        # First all images need to be transformed
        self.transform_all(image)

        # Step 1: Separate masks based on their suffixes
        contour_masks = {}
        for key in self.reg_data:
            if key.startswith(self.contour_identifier):
                suffix = key.split('_')[-1]
                if suffix not in contour_masks:
                    contour_masks[suffix] = []
                contour_masks[suffix].append(key)
    
        # Step 2: Combine masks with the same suffix using the '|' operator
        combined_masks = {}
        for suffix, masks in contour_masks.items():
            combined_mask = None
            for mask_key in masks:
                mask_image = self.reg_data[mask_key].astype(bool)  # Convert to bool data type
                if combined_mask is None:
                    combined_mask = mask_image
                else:
                    combined_mask |= mask_image
            combined_masks[f"image_{suffix}"] = combined_mask.astype(int)  # Convert back to int data type
    
        # Step 3: Combine the results using the '&' operator along the channel axis (axis=2)
        final_mask = None
        for mask_image in combined_masks.values():
            if final_mask is None:
                final_mask = mask_image
            else:
                final_mask &= mask_image
    
        self.reg_data['common_region'] = final_mask    
        return final_mask

    def save(self,image,path,visualise=True):
        """
        Save the timelapse series data and analysis results.

        Parameters:
            image (str): The name/key of the image used as a reference for saving data.
            path (str): The path where data and analysis results will be saved.
        """
        path = os.path.join(path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.common_region(image)
        df = pd.concat(self.analysis_results)
        
        df.to_csv(os.path.join(path,self.name+'.csv'))
        if visualise:
            dict_to_vtkFile(self.reg_data, os.path.join(path,self.name+'.vti'))

    
    def get_image(self, image_name: str) -> str:
        """
        Get the key of an image in the data dictionary.

        Returns:
            str: Key of the image.
        """
        return '_'.join([self.image_identifier, image_name])

    
    def get_contour(self, contour_name: str, image_name: str) -> str:
        """
        Get the key of a contour in the data dictionary.

        Returns:
            str: Key of the contour.
        """
        return '_'.join([self.contour_identifier, contour_name, self.get_image(image_name)])

    
    def get_all_images(self) -> List[str]:
        """
        Get the keys of all images in the data dictionary.

        Returns:
            list of str: List of image keys/names.
        """
        return [
            key[len(self.image_identifier) + 1:] 
            for key in self.data.keys() 
            if key.startswith(self.image_identifier)
        ]

    def get_all_contours(self) -> List[str]:
        """
        Get the keys of all contours in the data dictionary.

        Returns:
            list of str: List of contour keys/names.
        """
        return [
            self.contour_identifier + key[len(self.contour_identifier):] 
            for key in self.data.keys() 
            if key.startswith(self.contour_identifier)
        ]

    
    def get_all_data(self) -> List[str]:
        """
        Get all keys (images and contours) stored in the data dictionary.

        Returns:
            list of str: List of all keys.
        """
        return list(self.data.keys())

    
    def get_contours_from_image(self, image_name: str) -> List[str]:
        """
        Get the image and associated contour keys for a given image name.

        Parameters:
            image_name (str): The name/key of the image to retrieve.

        Returns:
            list of str: A list containing the associated contour keys.
        """
        contours = [
            self.contour_identifier + key[len(self.contour_identifier):] 
            for key in self.data.keys() 
            if key.startswith(self.contour_identifier) and (key.find(image_name) != -1)
        ]
        if len(contours) < 1:
            raise ValueError("Image not associated with any contour.")
        return contours

    
    def load_aim(self,path,crop=False):
        """
        Load image and position data from an AIM file.

        Parameters:
            path (str): The path to the AIM file.
            crop (bool): If True, crop images to the smallest bounding box (default: False).

        Returns:
            np.ndarray: The image data loaded from the AIM file.
            dict: The position information loaded from the AIM file.
        """
        data = aim.load_aim(path)
        image = np.asarray(data.data)
        position = data.position
        voxelsize = data.voxelsize.to('mm').magnitude[0]
        
        # This is to rescale different resolutions (rare case)
        if self.resolution is not None:
            if abs(self.resolution-voxelsize)>1e-3:
                image = self.rescale_image(image, res_from=voxelsize, res_to=self.resolution)
                voxelsize = self.resolution

        self.voxelsize=voxelsize

        if crop:
            if len(np.unique(image)>2):
                binary_mask = getLargestCC(combined_threshold(image))
            else:
                binary_mask = image
            # Crop image to smallest bounding box
            bb = boundingbox_from_mask(binary_mask)
            image = image[bb]
            position = update_pos_with_bb(position, bb)
        
        return image, position

    
    def rescale_image(self, data, res_from=1, res_to=1, order=1):
        """
        Rescale the image data to a different resolution.

        Parameters:
            data (np.ndarray): The image data to be rescaled.
            res_from (float): The original resolution of the image (default: 1).
            res_to (float): The desired resolution to which the image should be rescaled (default: 1).
            order (int): The order of interpolation used for rescaling (default: 1).

        Returns:
            np.ndarray: The rescaled image data.
        """  
        # Calculate the scaling factor for each axis
        scaling_factors = [res_from / res_to,]*3
        
        # Perform the upscaling using linear interpolation (order=1)
        return zoom(data, scaling_factors, order=order, mode='nearest')
    

    def update_size_and_position(self):
        """
        Update the size and position information for all images and contours to match the overall shape.
        """
        for key in self.get_all_data():
            self.data[key], self.position[key] = pad_array_centered(
                self.data[key], self.position[key], self.shape)



