import warnings
import os
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output
from ._image_overlay import _overlay_RGB_texture, overlay_mask_raw
from .interactive_gui_elements import updatable_renderer,data_container
from ipywidgets import interact, interactive, fixed, interact_manual, Dropdown
from ipywidgets import Layout, Button, Box, VBox, GridBox, Output
from ipywidgets import Button, GridBox, ButtonStyle
from .data_management import data_manager

from .io import File


planes = ['XY','XZ', 'YZ']
warnings.filterwarnings('ignore')
plane_to_axis = {'YZ':0,'XZ':1,'XY':2}


def extract_slice( image, slice_number ,plane):
    '''
    Where the magic happens
    Function accepts a dataset name, the slice number and the plane.
    It checks the bounds and returns the slice and the updated slice number if it is OOB
    '''
    if plane == 'YZ':
        if slice_number >= image.shape[0]:
            slice_number=image.shape[0]-1
        return image[slice_number, :, :].swapaxes(0,1),slice_number
    if plane == 'XZ':
        if slice_number >= image.shape[1]:
            slice_number= image.shape[1]-1
        return image[:, slice_number, :].swapaxes(0,1),slice_number
    if plane == 'XY':
        if slice_number >= image.shape[2]:
            slice_number= image.shape[0]-1
        return image[:, :, slice_number].swapaxes(0,1), slice_number

def axis_limit(images,plane):
    '''
    returns the maximum length of a single axis
    '''
    max_len = 0
    for key, img in zip(images.keys(), images.values()):
        current_len = img.shape[plane_to_axis[plane]]
        if current_len > max_len:
            max_len = current_len
    return max_len

def load_list_of_files(**kwargs):
    files = kwargs['files']
    datasets = kwargs.get('datasets',None)
    names=[]
    data=[]
    for file in files:
        n,d = load_file(file=file,datasets=datasets)
        names+=n
        data += d
    return names,data

def load_file(**kwargs):
    file = kwargs.get('file')
    datasets = kwargs.get('datasets',None)
    file_name_short = os.path.basename(os.path.splitext(file)[0])
    names=[]
    data=[]
    with File(file) as io_file:
        if datasets is None:
            datasets = io_file.get_dataset_names()
        for d in datasets:
            dat,meta = io_file.read(d)
            try:
                data.append(dat)
                names.append(file_name_short+" ['"+d+"']")
            except:
                pass

    return names,data

class jupyter_browser:
    '''
    A slice browser which can be used in Jupyter notebooks. The following code will
    allow you to see two 3d datasets.

    By right clicking in the images multiple images are selectable. There are also a few colormaps available
    it easy to add more.

    import matplotlib as plt
    import matplotlib
    import numpy as np
    %matplotlib notebook
    
    # Changes the size of the figure to (width, height) in inches
    matplotlib.rcParams['figure.figsize'] = (16,5)

    from ifb_framework.slice_browser import jupyter_browser
    jupyter_browser({'random':np.random.uniform(size=(30,40,50)),'ordered':np.arange((30*40*50)).reshape((30,40,50))})
    
    '''
    def __init__(self, *argv, **kwargs):
        '''
        can be created with:
        1. a dict of images
        2. a file name
        3. a list of files

        as keyword: 'datasets' a list of datasets can be given, this limits what is loaded from the files.
        as keyword: 'colormaps' a new set of colormaps can be given.
        as keyword: 'slice' can open on a specific slice
        as keyword: 'plane' opens with different plane
        as keyword: 'default_cmap' will use a different default cmap, must be in list


        '''
        self.data_container = data_container()

        if isinstance(argv[0],dict):
            self.data_container.add_data(list(argv[0].keys()),argv[0].values())
            
        elif isinstance(argv[0],list):
            if isinstance(argv[0][0],str):
                datasets = kwargs.get('datasets',None)
                self.data_container.add_data_with_handler(load_list_of_files,files=argv[0],datasets=datasets)
 
        elif isinstance(argv[0],str):
            datasets = kwargs.get('datasets',None)
            self.data_container.add_data_with_handler(load_file,file=argv[0],datasets=datasets)
            
        else:
            raise TypeError(f"The slice browser accepts a dict of images, a file name, or list of file names. You provided a {type(argv[0])}.")

        self.slice=kwargs.get('slice',0)
        self.plane=kwargs.get('plane','XY')

        self.active_images = self.data_container.provide_selected()

        self.slices = {name:extract_slice(self.active_images[name],self.slice,self.plane)[0] for name in self.active_images.keys()}
        self.figure = updatable_renderer()
        
        self.colormap_list = kwargs.get('colormaps',['gray','jet','plasma'])
        self.show_colorbar = True
        self.cbar=None

        self.default_cmap = kwargs.get('default_cmap',self.colormap_list[0])
        if self.default_cmap not in self.colormap_list:
            self.default_cmap=self.colormap_list[0]
        
        self.figure.render(self.slices,cmap=self.default_cmap, data_range=self.data_container.provide_data_ranges())
        
        self.make_gui() #must be last!!!


    def make_gui(self):
        '''
        This could easily be in __init__() but it's nice to separate.
        '''
   
        slice_max = axis_limit(self.active_images,self.plane) - 1
        self.Slice_slide = widgets.IntSlider(max=slice_max)
        self.Slice_slide.observe(self.change_slice, 'value')

        self.axis_select=widgets.Dropdown(
        options=['XY', 'XZ', 'YZ'],
        value='XY',
        description='Axis:',
        disabled=False)
        self.axis_select.observe(self.change_plane,'value')

        self.cmap_select=widgets.Dropdown(
            options=self.colormap_list,
            value=self.colormap_list[0],
            description='colormap:',
            disabled=False)
        self.cmap_select.observe(self.change_cmap,'value')
        
        self.data_container.w.observe(self.change_active_image, 'value')
        
        flip_order_button = widgets.Button(description="Flip image order", layout=Layout(width='auto'))
        def flip_order(b):
            '''
            Flips the order of images, for example in case you have a mask before an image instead of after.
            (they are passed psitionally)
            '''
            temp_selected = self.data_container.w.value
            self.data_container.w.options = self.data_container.w.options[::-1]
            self.data_container.w.value = temp_selected[::-1]
        flip_order_button.on_click(flip_order)

        self.close_counter = 0
        close_button = widgets.Button(description="Close browser", button_style='danger', icon='window-close', layout=Layout(width='auto'))
        self.close_warning = widgets.Output()
        def close_widget(b):
            '''
            Kill button for GUI since it likes to hog memory.
            '''
            if not self.close_counter:
                with self.close_warning:
                    print("Click again to kill browser")
                self.close_counter = 1
            else:
                self.data_container.delete()
                self.figure.delete()
                self.cmap_select.__del__()
                self.axis_select.__del__()
                self.Slice_slide.__del__()
                self.buttons.children[0].__del__()
                self.buttons.children[1].__del__()
                self.image_pane.__del__()
                self.close_warning.__del__()
                self.memory_message.__del__()
                self.slices = []
                clear_output()
                
        close_button.on_click(close_widget)
        
        self.memory_message = widgets.Output()
        with self.memory_message:
            print("Please click the 'Close browser' button when finished to conserve memory.")

        box_layout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    width='70%')
        self.box_control = VBox(children=[self.Slice_slide,self.axis_select,self.cmap_select], layout=Layout(width='auto',grid_area='navi'))
        self.data_container.w.layout = Layout(grid_area='select')
        self.buttons = VBox(children=[flip_order_button, close_button, self.close_warning], layout=Layout(width='100%', grid_area='button'))
        self.image_pane = Box(children=[self.figure.widget()], layout=Layout(width='auto',grid_area='image'))
        self.memory_message.layout = Layout(grid_area='memmsg')
        self.figure.fig.suptitle('Best Slice Browser Ever')
        display(GridBox([self.image_pane,self.box_control,self.data_container.w, self.buttons, self.memory_message],
        layout=Layout(
            grid_template_rows='auto auto auto auto auto',
            grid_template_columns='33% 33% 33%',
            grid_template_areas='''
            "image image image"
            "image image image"
            "image image image"
            ".    select navi"
            "memmsg button navi"
            ''')))
        #self.update_figure()
            

    def change_slice(self,change):
        self.slice=change['new']
        self.update_figure()

    def change_plane(self,plane):
        self.plane=plane['new']
        self.slice=0
        slice_max = axis_limit(self.active_images,self.plane) - 1
        self.Slice_slide.max=slice_max
        self.Slice_slide.value=0
        self.update_figure()

    def colorbar_crap(self):
        pass
#         if self.cbar is not None:
#             try:
#                 self.cbar.remove()
#             except:
#                 pass
#         self.cbar=self.figure.colorbar(self.plot_obj)
#         unit = give_units(self.active_image[0])
#         if unit is not None:
#             self.cbar.ax.set_ylabel(unit, rotation=270)

    def choose_overlay(self):
        self.overlay_choice=widgets.Select(
            options=['side by side', 'overlay RGB', 'transparent'],
            value='side by side',
            description='Multi-image display mode:',
            disabled=False
        )
        self.overlay_choice.observe(self.overlay_options,'value')
        self.box_control.children =tuple(list(self.box_control.children ) + [self.overlay_choice])

    def clear_overlay_widget(self):
        self.box_control.children = tuple(list(self.box_control.children)[:-1])

    def update_figure(self):
        self.close_counter = 0
        self.close_warning.clear_output()
        self.slices = {name:extract_slice(self.active_images[name],self.slice,self.plane)[0] for name in self.active_images.keys()}
        self.figure.render(self.slices,cmap=self.cmap_select.value, data_range=self.data_container.provide_data_ranges())

    def change_active_image(self,value):
        self.active_images = self.data_container.provide_selected()
        if len(value['new']) > 1 and len(value['old']) <= 1:
            self.choose_overlay()
        if len(value['new']) <= 1 and len(value['old']) > 1:
            self.clear_overlay_widget()
            self.figure.change_renderer(None)
        self.update_figure()

    def change_cmap(self,cmap):
        self.update_figure()

    def overlay_options(self,value):
        if value['new']=='side by side':
            self.figure.change_renderer(None)
            
        if value['new']=='overlay RGB':
            self.show_colorbar = False
            self.figure.change_renderer(_overlay_RGB_texture)

        if value['new']=='transparent':
            self.show_colorbar = False
            self.figure.change_renderer(overlay_mask_raw)
        
        self.update_figure()

        


    