import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import inspect
from ipywidgets import Layout, Button, Box, VBox, HBox, GridBox, Output, FloatSlider

class data_container:
    '''
    The following is a class for conveniently storing data to be used with a GUI in a dict with 
    some methods, hence the name. Comes with own selection widget.
    
    Attributes
    ----------
    data : dict
        Data stored in the container with names
    selection : tuple
        Current selection in widget
    height : {'Npx', 'N%', 'auto'} where N is an int
        Height of the selection widget.
    width : {'Npx', 'N%', 'auto'} where N is an int
        Width of the selection widget.
    description : str
        Description of selection widget
    w : :any:ipywidgets.SelectMultiple
        Widget used to select data in container
    custom_observe : function or None
        Function to automatically be called when the widget value is changed.
        Takes new selection as input.
    
    '''
    def __init__(self, height='100px', width='400px', description='', custom_observe=None):
        '''
        Parameters
        ----------
        height : {'Npx', 'N%', 'auto'} where N is an int
            Height of the selection widget.
        width : {'Npx', 'N%', 'auto'} where N is an int
            Width of the selection widget.
        description : str
            Description of selection widget
        custom_observe : function
            Function to automatically be called when the widget value is changed.
            Takes new selection as input.
        '''
        self.data={}
        self.selection= None
        self.height = height
        self.width = width
        self.description = description
        self.w = widgets.SelectMultiple(
            description=description,
            disabled=False,
            layout=Layout(height=height,width=width)
        )
        self.w.observe(self.observe,'value')
        self.custom_observe=custom_observe
        self.axis_converter = {'YZ':0,'XZ':1,'XY':2}
        self.data_range = None
        
    def add_data(self, names, data):
        '''
        Add data to data_container
        '''
        val = self.w.value
        for name,d in zip(names,data):
            self.data[name]=d
        self.w.options=self.data.keys()
        if val:
            self.w.value=val
        else:
            self.w.value=[names[0]]
            
        self.selection= self.data.keys() #MW added this to show all data 

    def add_data_with_handler(self, input_handler, **kwargs):
        '''
        Add data to data_container after passing to custom handler
        '''
        names,data = input_handler(**kwargs)
        self.add_data(names,data)

    def provide_selected(self):
        '''
        Give dict of current selections in widget (with actual data)
        '''
        return {d:self.data[d] for d in self.selection}
    
    def observe(self,value):
        '''
        Function triggered when widget selection is changed. Can have custom handler.
        Should not be used externally (private func - but can't add underscore)
        '''
        self.selection = list(value['new'])
        if self.custom_observe is not None:
            self.custom_observe(self.provide_selected())
            
    def copy(self):
        '''
        Makes a second copy of the widget with current data (separate from first - not referenced)
        '''
        new_copy = data_container(height=self.height, width=self.width, description=self.description, custom_observe=self.custom_observe)
        new_copy.add_data(list(self.data.keys()), self.data.values())
        return new_copy
    
    def axis_limit(self,plane):
        '''
        Gives maximum axis length of data (only works for 3D currently)
        '''
        return self.data.shape[axis_converter[plane]]-1
    
    def get_data_ranges(self):
        '''
        Gives the maximum and minimum data points for data in the container. Returns
        a nested list
        '''
        ranges = {}
        for name in self.data:
            min = np.min(self.data[name].data)
            max = np.max(self.data[name].data)
            ranges[name] = [min, max]
        return ranges
    
    def provide_data_ranges(self):
        '''
        Gives back a nested list of the data ranges for the active images
        '''
        if self.data_range is None:
            self.data_range = self.get_data_ranges()
            
        data_ranges = []
        names = self.provide_selected().keys()
        for name in names:
            data_ranges.append(self.data_range[name])
        return data_ranges
    
    def delete(self):
        '''
        Deletes important contents of class to free up memory
        '''
        self.w.__del__()
        self.data = {}
        

    
    
class updatable_renderer:
    '''
    This class is meant to eliminate (or at least reduce) the hassle of working wit matplotlib figures,
    especially in the case where they are being updated frequently (such as in a GUI).
    
    Attributes
    ----------
    fig : matplotlib figure object
        The figure object containing all plots
    plot_function : function
        The current plotting function to use
    '''
    def __init__(self, images=None, plot_function=None):
        '''
        Parameters
        ----------
        plot_function : function
            The current plotting function to use
        images : list (or nested list) of numpy arrays
            Images to pass to render() on class initiation
        '''
        self.plot_function = plot_function
        self.fig = None
        self.data_range = None
    
        if images is not None:
            self.fig=self.render(images, return_fig=True)
            
    def change_renderer(self, plot_function):
        '''
        Change the plotting function to use
        '''
        self.plot_function=plot_function
        
    def widget(self):
        '''
        Returns the figure canvas, which can be treated as a widget (in ipywidgets for example)
        '''
        if self.fig:
            return self.fig.canvas
        
    def delete(self):
        '''
        Deletes main class components, to free memory
        '''
        self.fig.clear()
        self.fig.canvas.__del__()
        
    def render(self, in_images, fig=None, plot_function=None, custom_renderer=None, file_name=None, return_fig=True, cmap=None, orientation='horizontal', data_range=[[None,None]], **kwargs):
        '''
        This is the bread and butter of the updatable_renderer class. It takes images, a plot function,
        or even a custom rendering function, along with various arguments and determines whether the current
        figure can be updated or needs to be redrawn - then does so. Updating figures instead of redrawing
        saves a significant amount of processing and is handy for quickly updating plots.
        
        Parameters
        ----------
        in_images : dict
            The images to pass to plot_function
        fig : None or matplotlib figure object
            The figure to draw to. If None, checks to see if self.fig exists, otherwise
            draws a new one. Default None.
        plot_function : function
            Plotting function to be applied to in_images. Must return either a list of
            numpy arrays or a nested list (max one-level deep). Otherwise custom_renderer
            must be used. Default is None, which causes all images to be displayed normally
            side by side.
            In the case of the former, each array is plotted separately in its own
            axis, while with the latter all arrays in each sub-list are plotted to
            one axis (allowing, for example, mask overlays)
        custom_renderer : function
            NOT IMPLEMENTED. If this rendering function does not meet your needs you can 
            supply your own here.
        file_name : str
            NOT IMPLEMENTED. Saves the resulting figure to the file name.
        return_fig : bool
            Returns the new fig object (same as getting self.fig). Default True.
        cmap : matplotlib cmap
            The colourmap to use with the plot, if applicable. Default None.
        orientation : {'horizontal', 'vertical'}
            Orientation if multiple axes are drawn. Default horizontal
        data_range : [num, num] or [[num, num], [num, num]]
            The data range to use for the colourmap. Handy for keeping colours consistent
            across mutiple images. Default [None, None].
        '''
        
        if isinstance(in_images, list):
            temp_dict = {}
            for i, image in enumerate(in_images):
                temp_dict[i] = image
            in_images = temp_dict
        
        # Interactive must be OFF for redrawing figures, ON for updating them
        plt.ioff() 
        
        # Should be able to provide your own renderer, which skips all below
        if custom_renderer:
            raise NotImplemented("Custom renderer is only partially implemented, and may need refining.")
            return custom_renderer(**kwargs)
            
        # Flags for various redraw_conditions
        recreate_axis = False
        recreate_fig = False
        nested_images = False
        
        # If no figure given directly, take from updatable_renderer class
        if fig is None:
            fig = self.fig
        # If that does not exist either, create
        if fig is None:
            recreate_axis = True
            fig = plt.figure()
        # Interactive must be OFF for redrawing figures, ON for updating them
        plt.ion()
        
        # If no plot function is given directly, take from updatable_renderer class
        if plot_function is None:
            plot_function = self.plot_function
        # If that is still None, take images as given
        if plot_function is None:
            # Must convert to list
            out_images = list(in_images.values())
        else:
            # Pass images to function. Plot function must have 'images' kwarg and return either
            # a list of numpy arrays, or a nested list of arrays (one level deep)
            kwargs['images'] = list(in_images.values())
            out_images = plot_function(**kwargs)
            
        # Make sure the data range is the same (otherwise colormap issues)
        if self.data_range != data_range:
            recreate_axis = True
        
        # Define figure axes
        axarr=fig.axes
        # Check number of images returned by plot_function
        num_images = len(out_images)
        # If number of plots does not match output of plot_function, redraw fig
        if len(axarr)!=num_images:
            recreate_fig = True
            # If out_images is a nested list, take note
            for image in out_images:
                if isinstance(image, list):
                    nested_images = True
        else:
            for i, image in enumerate(out_images):
                # If out_images is a nested list, take note
                if isinstance(image, list):
                    nested_images = True
                    # If the number of images per axis is not the same as before, redraw
                    if len(axarr[i].images) != len(image):
                        recreate_fig = True
                    # If any sub-image shape mismatches the axis, redraw
                    for sub_image in image:
                        if not np.array_equal(axarr[i].images[0].get_array().shape, sub_image.shape[:2]): # Note that 3D imagea are allowed here, as in the case of mask overlays
                            recreate_fig = True
                # If images shape mismatches the axes, redraw
                elif not np.array_equal(axarr[i].images[0].get_array().shape, image.shape):
                    recreate_fig = True
                # If previous axes had nested images but we don't now, redraw
                elif len(axarr[i].images) != 1:
                    recreate_fig = True
        # NOTE: I'm fairly sure some of these recreate_figs can be replaced with recreate_axes, which might
        # save a little processing time, but currently this works.
        
        # If the figure must be redrawn
        if recreate_fig:
            # Interactive must be OFF for redrawing figures, ON for updating them
            plt.ioff()
            fig.clear()
            axarr.clear()
            if orientation == 'horizontal':
                gs=fig.add_gridspec(1, num_images)
                for i in range(num_images):
                    fig.add_subplot(gs[0,i])
            elif orientation == 'vertical':
                gs=fig.add_gridspec(num_images, 1)
                for i in range(num_images):
                    fig.add_subplot(gs[i,0])
            else:
                raise ValueError(f"Orientation was given as {orientation}, it must be 'vertical' or 'horizontal'")
            recreate_axis = True
            axarr=fig.axes
            plt.ion()

        for i, (image, image_name) in enumerate(zip(out_images, in_images.keys())):
            axarr[i].title.set_text(image_name)
            # If axis must be redrawn, we imshow
            if recreate_axis:
                # If we have multiple images per axis, imshow multiple times per axis
                if nested_images:
                    for sub_image in image:
                        axarr[i].imshow(sub_image, cmap=cmap, vmin=data_range[i][0], vmax=data_range[i][1])
                else:
                    axarr[i].imshow(image, cmap=cmap, vmin=data_range[i][0], vmax=data_range[i][1])
            # If no flags were raised, go ahead and update the current figure and axes with the new data
            else:
                if nested_images:
                    for j, sub_image in enumerate(image):
                        axarr[i].images[j].set_data(sub_image)
                        axarr[i].images[j].set_cmap(cmap)
                else: 
                    axarr[i].images[0].set_data(image)
                    axarr[i].images[0].set_cmap(cmap)

        self.fig = fig
        self.data_range = data_range
        
        if return_fig:
            return fig

        if file_name:
            raise NotImplementedError("Saving plot to file is not yet implemented, but should be doable in the matplotlib ipympl interface.")

    def register_mouse_clicks(self, function):
        self.clicks = []
        self.click_link = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.click_func = function
            
    def _on_click(self, event):
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None:
            return
        self.clicks.append([ix,iy])
        self.clicks = self.click_func(self.clicks, self.fig, event)
        
    def unregister_mouse_clicks(self):
        self.clicks = []
        self.fig.canvas.mpl_disconnect(self.click_link)
        
    def register_mouse_drags(self, function):
        self.clicks=[]
        self.drag_start = self.fig.canvas.mpl_connect('button_press_event', self._start_drag)
        self.drag_release = self.fig.canvas.mpl_connect('button_release_event', self._release_drag)
        self.drag_link = None
        
    def _start_drag(self,event):
        self.drag_link = self.fig.canvas.mpl_connect('motion_notify_event', self._on_drag)

    def _on_drag(self,event):
        ix, iy = event.xdata, event.ydata
        self.clicks.append([ix,iy])

    def _release_drag(self,event):
        self.fig.canvas.mpl_disconnect(self.drag_link)
        
    def unregister_mouse_drags(self):
        self.clicks = []
        self.fig.canvas.mpl_disconnect(self.drag_start)
        self.fig.canvas.mpl_disconnect(self.drag_release)
        
        
class slider_box:
    '''
    Partial implementation of easy class for making widget sliders 
    '''
    def __init__(self,names,slidertype=FloatSlider,box=VBox,custom_observe=None,**kwargs,):
        self.slide ={name:slidertype(description=name) for name in names }
        self.box = box(list(self.slide.values()))
        self.custom_observe = custom_observe

    def observe(self):
        pass