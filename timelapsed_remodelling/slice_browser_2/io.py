import itk
import os
import numpy as np
from ifb_framework.timelapsed_remodelling.visualise import vtkFile_to_dict

class File:
    '''
    Simple class that uses itk.imread for reading image files.
    '''

    def __init__(self, name):
        '''
        Constructor for the SimpleFile class.

        Parameters
        ----------
        name : str
            Name of the file to read.
        '''
        self.name = name

    def read(self,name):
        '''
        Method to read an image from the file.

        Returns
        -------
        image : itk.Image
            The image read from the file.
        '''
        if not os.path.exists(self.name):
            raise FileNotFoundError("File '{}' does not exist.".format(self.name))

        if self.name.endswith('.vti'):
            rotated_array = vtkFile_to_dict(self.name)
        else:
            original_array = itk.GetArrayFromImage(itk.imread(self.name))
            rotated_array = np.transpose(original_array, (2, 1, 0))
        
        return rotated_array, None

    def __enter__(self):
        '''
        Method for the context manager that allows using the `with` statement.
        '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Method called when exiting the context manager.
        '''
        pass  # No cleanup needed in this case, so we leave it empty

    def get_dataset_names(self):
        return ['aim']
       
    def _separate_unit_from_dataset_attributes(self, name):
        return 'unitless'

    def get_dataset_unit(self, name):
        return 'unitless'

    def get_dataset_attributes(self, name):
        return 'unitless'












