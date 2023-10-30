import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb



def overlay_mask(image, mask, mask_color='red', mask_alpha=0.3, ax=None, return_objects=False, update=False):
    """
    Plot a bone image with a mask overlayed.

    Parameters
    ----------
    image : 2d-array
        The bone image to display
    mask : bool 2d-array
        The mask to overlay over the bone image
    mask_color : string or tuple
        A color as given to any matplotlib function. If given
        as an RGBA color, the alpha will be silently ignored.
    mask_alpha : float
        The alpha value used for overlaying the mask. 1 meaning
        fully opaque, 0 meaning fully transparent.
    ax : matplotlib axis
        If given, will plot to the specified
        axis. Otherwise, will plot to the default
        axis. Furthermore, if not given, the plot will be shown
        automatically. Otherwise, the function returns
        without showing the plot (useful for sub-plots).

    """
    show_immediately = False
    if ax is None:
        show_immediately = True
        ax = plt

    color = to_rgb(mask_color) + (mask_alpha,)
    mask = mask[:,:,np.newaxis] * color
    
    if update:
        I = update[0]
        M = update[1]
        I.set_data(image)
        M.set_data(mask)
    else:
        I = ax.imshow(image, cmap='bone')
        M = ax.imshow(mask)

    if show_immediately:
        plt.show()
    if return_objects:
        return(ax,I,M)
    
def overlay_mask_raw(images, mask_color='red', mask_alpha=0.3):
    '''
    Returns raw arrays instead of fig. Made to be used with GUI toolbox renderer
    '''
    color = to_rgb(mask_color) + (mask_alpha,)
    mask = images[1][:,:,np.newaxis] * color
    image = images[0]
    return [[image,mask]] # Nested list indicates to renderer that images are on one plot
    
def overlay_RGB(images, file_name=None,legend=True,fig=None,return_fig=False):
    '''
    Plots the overlay of always two consecutive grayscale images
    in the list of images passed to this function. For example
    if two images are passed to this function, the first image
    will occupy the color [1.0, 0.0, 0.5] (RGB) and the second image
    will occupy the color [0.0, 1.0, 0.5] (RGB). This means that
    if a pixel is only white in the first image, it will appear
    red-bluish, and if it is only white in the second image it will
    appear green-bluish. If they are white in both images, the resulting
    color will also be white. For gray values in between you get a
    simple dimming effect of the respective color and some mixture
    of the final colors.

    This function can be useful to display image registration results
    as it will automatically highlight in color which pixels are different
    between the images while perfectly registered pixels will stay white.

    If more than two images are passed to this function, a panel
    with several plots will be generated.

    Parameters
    ----------
    images : list like
        List of images to overlay
    file_name : str, optional
        Instead of opening a window will save to file

    '''
    create_all= False
    if fig in None:
        create_all = True
        fig = plt.figure()

    axarr=fig.axes
    overlays = _overlay_RGB_texture(images)
    num_images = len(overlays)


    if axarr or len(axarr)!=num_images:
        fig.clear()
        gs=fig.add_gridspec(1, num_images)
        for i in range(num_images):
            fig.add_subplot(gs[0,i])
        create_all = True
        axarr=fig.axes
    
    for i, overlay in enumerate(overlays):
        if create_all: #if it's obvious this won't work
                axarr[i].imshow(overlay)
        else: #try and update axis
            if  np.array_equal(axarr[i].images[0].get_array().shape,overlay):#check size
                axarr[i].images[0].set_data(overlay)
            else:
                axarr[i].imshow(overlay)

                
    if legend:
        colors = [[1., 0., 0.5, 1.], [0., 1., 0.5, 1.], [1., 1., 1., 1.]]
        labels = ['resorbed', 'formed', 'quiescence']
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0.)

    if return_fig:
        return fig

    if file_name:
        raise NotImplementedError('Saving to file for the overlay_RGB is not implemented, yet.')
    else:
        fig.show()

def _overlay_RGB_texture(images, red_green_scheme=False):
    '''
    Plots the overlay of always two consecutive grayscale images
    in the list of images passed to this function. For example
    if two images are passed to this function, the first image
    will occupy the color [1.0, 0.0, 0.5] (RGB) and the second image
    will occupy the color [0.0, 1.0, 0.5] (RGB). This means that
    if a pixel is only white in the first image, it will appear
    red-bluish, and if it is only white in the second image it will
    appear green-bluish. If they are white in both images, the resulting
    color will also be white. For gray values in between you get a
    simple dimming effect of the respective color and some mixture
    of the final colors.

    This function can be useful to display image registration results
    as it will automatically highlight in color which pixels are different
    between the images while perfectly registered pixels will stay white.

    If more than two images are passed to this function, a panel
    with several plots will be generated.

    Parameters
    ----------
    images : list like
        List of images to overlay
    file_name : str, optional
        Instead of opening a window will save to file
    red_green_scheme : boolean
        If true, set blue component to zero, giving red, green, and yellow
        as overlay colours. Looks a little nicer for binary images. Default
        false.

    '''

    for image in images:
        if image.ndim != 2:
            error_message = 'Image passed to overlay_RGB has dimension {}, but all images must have dimension 2.'.format(image.ndim)

    for shape in [image.shape for image in images[1:]]:
        if shape != images[0].shape:
            error_message = 'Images passed to overlay_RGB need to have the same size. Sizes passed: {}.'.format([image.shape for image in images])
            print(error_message)
            raise ValueError(error_message)

    num_images = len(images)-1

    overlays=[]
    for i, (im1, im2) in enumerate(zip(images[:-1], images[1:])):
        overlay = np.zeros((im1.shape) +(3,),dtype=float)
        overlay[:,:,0] = im1.astype(float)
        overlay[:,:,1] = im2.astype(float)
        if red_green_scheme:
            overlay[:,:,2] = 0
        else:
            overlay[:,:,2] = im1.astype(float)*0.5 + im2.astype(float)*0.5
        overlay-=np.min(overlay)
        overlay /= np.max(overlay)
        overlay[overlay < 0] = 0.
        overlays.append(overlay)


    return overlays
