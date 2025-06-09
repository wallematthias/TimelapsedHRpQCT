from timelapsed_remodelling.visualise import _overlay_RGB_texture, overlay_mask
import os
import numpy as np

from .io import File

axis_converter = {'YZ':0,'XZ':1,'XY':2}
class meta_image:
    '''
    A class for wrapping diverse 3d datasets with some visualisation meta data
    '''
    def __init__(self,data,meta_data):
        self.data = data
        self.meta_data = meta_data
        self.colormap = None
        self.pipline = []
        self.vmax = np.max(data)
        self.vmin = np.min(data)

    def axis_limit(self,plane):
        return self.data.shape[axis_converter[plane]]-1
    
class renderable_slice:
    '''
    A class which defines the "standard" api for LBB slice browsing programs. Class can contain any data be constructed in anyway
    Must have a method called render which can operate on a matplotlib axes.
    '''
    def __init__(self,slice_data,colormap, slice_number):
        self.slice = slice_data
        self.colormap = colormap
        
    
    def render(self,axes,vmin=None,vmax=None):
        if len(self.slice.shape)!=2:
            vmax=0
            vmin=0
        return axes.imshow(self.slice,cmap=self.colormap, interpolation = 'nearest',vmin=vmin,vmax=vmax)

class renderable_slice_with_mask:
    '''
    An renderable slice which uses a different render method than existing version.
    '''
    def __init__(self,slice_data):
        self.slice = slice_data[0]
        self.mask = slice_data[1]
        self.colormap = None
        
    
    def render(self,axes,vmin=None,vmax=None):
        overlay_mask(self.slice,self.mask,ax=axes)
        return None


class data_manager:
    '''
    This is like the worst class ever. It totaly needs to be broken up and relevant parts added to metaImage.
    It is the backend which hosts all the data for the slice browser. It can return various slices and information
    about the colume
    '''

    def __init__(self):

        self.images={}
        self.slice_method = self.provide_single_image
        self.default_cmap = None

    def set_default_cmap(self,cmap):
        self.default_cmap=cmap
        for k,d in self.images.items():
            d.colormap=cmap

    def addImage(self,path,datasets=None):
        '''
        Function for loading files from the disk. It is really stupidly names, should really be "load_file".
        Probably only relevant for the QT5 standalone slicer.
        '''
        filename, file_extension = os.path.splitext(path)
        name=os.path.basename(filename)

        if file_extension.lower() is '.h5i':
            with File(path) as io_file:
                if datasets==None:
                    datasets=io_file.get_dataset_names()

                for dataset in datasets:
                    image_data,meta_data = io_file.read(dataset)
                    self.images[name+'.'+dataset] = meta_image(image_data,meta_data)

        if file_extension.lower() == '.aim':
            with File(path) as io_file:
                image_data,meta_data = io_file.read('')
                self.images[name] = meta_image(image_data,meta_data)


    def load_list_of_files(self,files,datasets=None):
        for file in files:
            self.load_file(file,datasets=datasets)

    def load_file(self,file,datasets=None):
        file_name_short = os.path.basename(os.path.splitext(file)[0])
        with File(file, 'r') as io_file:
            if datasets is None:
                datasets = io_file.get_dataset_names()
            for d in datasets:
                data,meta = io_file.read(d)
                self.images[file_name_short+" ['"+d+"']"] = meta_image(data,meta)

    def add_data(self,data):
        '''
        Function which adds more data to a slice browser. Don't know if it's useful post construction...

        data must be a dict
        '''
        for name,image_data in data.items():
            self.images[name] = meta_image(image_data,None)

    def value_range(self,images):
        '''
        Returns the max and min values in all selected images.

        TODO: If the images use diverse units it should return the max and mins per images or maybe per unit...
        '''
        ranges=[]

        for i in images:
            ranges.append([self.images[i].vmin,self.images[i].vmax])

        if len(images)==1:
            vmin = ranges[0][0]
            vmax = ranges[0][1]
        else:
            #print(ranges)
            vmin = None#np.min(np.array(ranges)[:,0])
            vmax = None#np.max(np.array(ranges)[:,1])
        return vmin,vmax

    def give_units(self,image):
        '''
        FUnction which returns the units of a stored dataset, very useful for colorbars
        '''
        if hasattr(self.images[image].data,'units'):
            return str(self.images[image].data.units)
        else:
            return None

    def extract_single_slice(self, image, slice_number ,plane):
        '''
        Where the magic happens
        Function accepts a dataset name, the slice number and the plane.
        It checks the bounds and returns the slice and the updated slice number if it is OOB
        '''
        if plane == 'YZ':
            if slice_number >= self.images[image].data.shape[0]:
                slice_number=self.images[image].data.shape[0]-1
            return self.images[image].data[slice_number, :, :].swapaxes(0,1),slice_number
        if plane == 'XZ':
            if slice_number >= self.images[image].data.shape[1]:
                slice_number=self.images[image].data.shape[1]-1
            return self.images[image].data[:, slice_number, :].swapaxes(0,1),slice_number
        if plane == 'XY':
            if slice_number >= self.images[image].data.shape[2]:
                slice_number=self.images[image].data.shape[0]-1
            return self.images[image].data[:, :, slice_number].swapaxes(0,1), slice_number

    def provide_single_image(self,image,slice_number,plane):
        '''
        Returns a slice from a single dataset
        '''
        slice_data,slice_number = self.extract_single_slice(image[0],slice_number,plane)
        return renderable_slice(slice_data, self.images[image[0]].colormap,slice_number)

    def provide_rgb_overlay(self,image,slice_number,plane):
        '''
        returns a slice from 2 datasets overlayed in rgb colour scheme
        '''
        slice_dataA,slice_number = self.extract_single_slice(image[0],slice_number,plane)
        slice_dataB,slice_number = self.extract_single_slice(image[1],slice_number,plane)
        slice_data=_overlay_RGB_texture([slice_dataA,slice_dataB])
        return renderable_slice(slice_data[0], self.images[image[0]].colormap,slice_number)

    def provide_mask_overlay(self,image,slice_number,plane):
        '''
        provides a slice with a transparent mask, agains needing two datasets
        '''
        slice_dataA,slice_number = self.extract_single_slice(image[0],slice_number,plane)
        slice_dataB,slice_number = self.extract_single_slice(image[1],slice_number,plane)
        return renderable_slice_with_mask((slice_dataA,slice_dataB))

    def give_a_slice(self,image,slice_number,plane):
        '''
        Not sure ehy this is here... I blame the coffee
        '''
        return self.provide_single_image(image,slice_number,plane)

    def axis_limit(self,image,plane):
        '''
        returns the maximum length of a single axis
        '''
        return self.images[image[0]].axis_limit(plane)

    def list_of_loaded_images(self):
        '''
        returns a list of all loaded images
        '''
        return list(self.images.keys())
        