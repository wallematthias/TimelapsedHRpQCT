import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np 
from . import custom_logger

def dict_to_vtkFile(data_dict, output_filename, spacing=None, origin=None, array_type=vtk.VTK_FLOAT):
    '''Convert numpy arrays in a dictionary to vtkImageData

    Default spacing is 1 and default origin is 0.

    Args:
        data_dict (dict):       A dictionary with keys as names and numpy arrays as values
        spacing (np.ndarray):   Image spacing
        origin (np.ndarray):    Image origin
        array_type (int):       Datatype from vtk

    Returns:
        vtkImageReader:         The corresponding vtkImageReader or None
                                if one cannot be found.
    '''
    # Set default values
    if spacing is None:
        spacing = np.ones_like(next(iter(data_dict.values())).shape)
    if origin is None:
        origin = np.zeros_like(next(iter(data_dict.values())).shape)

    # Convert data_dict to vtkImageData
    image = vtk.vtkImageData()
    for name, array in data_dict.items():
        temp = np.ascontiguousarray(np.atleast_3d(array))
        vtkArray = numpy_to_vtk(
            temp.ravel(order='F'),
            deep=True, array_type=array_type
        )
        image.SetDimensions(array.shape)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        vtkArray.SetName(name)  # Set the name of the data array
        image.GetPointData().AddArray(vtkArray)  # Add the data array to vtkImageData

    # Write the dataset to .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(image)
    writer.Write()

def vtkFile_to_dict(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    # Create an empty dictionary to store the NumPy arrays with their names
    image_data_dict = {}
    
    # Get the number of point data arrays (fields) in the VTK file
    num_arrays = reader.GetNumberOfPointArrays()
    
    for i in range(num_arrays):
        array_name = reader.GetPointArrayName(i)
        custom_logger.info(f"Reading dataset with array name: {array_name}")
    
        # Get the i-th dataset (point data array)
        array = reader.GetOutput().GetPointData().GetArray(i)
    
        if array is not None:
            # Convert the VTK data array to a NumPy array
            numpy_array = vtk_to_numpy(array)
    
            # Get the original dimensions of the dataset
            extent = reader.GetOutput().GetExtent()
            x_min, x_max, y_min, y_max, z_min, z_max = extent
            x_dim = x_max - x_min + 1
            y_dim = y_max - y_min + 1
            z_dim = z_max - z_min + 1
    
            # Reshape the NumPy array to its original shape
            original_array = numpy_array.reshape((z_dim, y_dim, x_dim))
            rotated_array = np.transpose(original_array, (2, 1, 0))
    
            # Add the NumPy array to the dictionary with the image name as the key
            image_data_dict[array_name] = rotated_array
        else:
            custom_logger.info("Error: Unable to read the dataset.")

    return image_data_dict