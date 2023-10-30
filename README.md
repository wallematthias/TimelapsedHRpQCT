# README for Stack Shift Artefact Correction Algorithm

This repository contains the implementation of a method for correcting stack shift artefacts in medical imaging, developed by Matthias Walle at the Institute for Biomechanics, ETH Zurich. This document provides an overview of the method and instructions on how to use the code.

## Method Steps
The method consists of the following steps:

1. Split scans into individual stacks consisting of 168 axial slices. The first stack is considered the most proximal stack in the image.

2. Align each stack across timepoints using rigid image registration. The registration of the first stack in the image is initialized by aligning the center of masses across timepoints. For the remaining stacks, the registration result of the prior stack is used as an initial guess.

3. Create a "super-stack" for each stack by superimposing the transformed images from all timepoints to a reference image. The reference image is the first scan acquired.

4. Register the super-stacks, starting with the most distal super-stacks. The registration region is limited to include only 15 slices on each adjacent super-stack border.

5. Combine the transformation matrices by a composite transform for each individual stack across each timepoint. The appropriate combinations of transformations for each stack are tracked using a graph model.

6. Fill gaps in the stack-corrected images using data from adjacent longitudinal scans, with preference given to the closer timepoints. If gaps are still present, fill remaining gaps using synthetic data generated using a greyscale closing kernel between the image stacks.

## Interpolation and Optimization
All registration is performed using rigid-body registration, with a Powell optimization to maximize the correlation coefficient between images within the periosteal bone contour. Greyscale images are transformed using linear interpolation and binary segmentations and masks are transformed using nearest-neighbor interpolation.

## Gap Filling
An initial small structuring kernel (3x3x3 voxels in size) is applied to maintain local detail in closing small gaps, followed by a slightly larger kernel (3x3x5 voxels in size) to fill more substantial gaps. Gaps of 5 voxels between stacks can be filled, where larger gaps are considered too substantial to fill synthetically without notable deviation from representing the true underlying bone microarchitecture.

## Installation
1. Open your terminal or command prompt.

2. Run the following command to install the package via pip:
pip install git+https://github.com/wallematthias/multistack_registration

3. If the package has any dependencies, pip will automatically download and install them for you.

4. Once the installation is complete, you will be able to use `multistack_registration` as command line funciton. With the following command line arguments.

    --input <str>
        Specifies the path to the input files. If not provided, the current folder is used as the default.

    --keyImage <str>
        Specifies the filename pattern for the image files to be registered. Both images need to have the same filename, except for the last letter which should indicate the timepoint (e.g., `NameA_T1.AIM`, `NameA_T2.AIM`, `NameA_T3.AIM`). To register all three files, the input pattern should be `NameA_T?`.

    --keyMask <str> ...
        Specifies the mask identifier to be used (e.g., `TRAB_MASK`, `CORT_MASK`). The mask filenames need to follow a specific pattern (`NameA_T1_TRAB_MASK.AIM`, `NameA_T1_CORT_MASK.AIM`, etc.) and be in the same folder as the key images.

    --transform <str>
        Specifies a custom path for the transformations to be saved.

    --output <str>
        Specifies the output path. If not provided, the default value is `'.'`.

    --stackHeight <int>
        Specifies the height of the stack. If not provided, the default value is `168`.

    --thresholds <int> ...
        Specifies the segmentation thresholds within the masks that were provided. If not provided, the default value is `None`.

    --options <str>
        Specifies the path to additional options. If not provided, the default value is `None`.

    For example:
        `multistack_registration --input '/path/to/input/files' --keyImage 'NameA_T?' --keyMask 'TRAB_MASK' 'CORT_MASK' --thresholds 320 450 --output '/path/to/output/files'`


5. If you need to upgrade or uninstall the package at a later time, you can use the following commands:
pip install --upgrade git+https://github.com/username/repo-name.git
pip uninstall package-name