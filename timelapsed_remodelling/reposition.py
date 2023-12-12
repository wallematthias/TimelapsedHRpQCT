import numpy as np
import numpy as np
import itertools
from . import custom_logger

def boundingbox_from_mask(mask, return_type='slice'):
    """Return tuple of slices or lists that describe the bounding box of the given mask.

    """
    if not np.any(mask):
        raise ValueError('Given mask is empty. Cannot compute a bounding box!')

    out = []
    try:
        for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
            nonzero = np.any(mask, axis=ax)
            extent = np.where(nonzero)[0][[0, -1]]
           # extent[1] += 1  # since slices exclude the last index
            if return_type == 'slice':
                out.append(slice(*extent))
            elif return_type == 'list':
                out.append(extent.tolist())
    except IndexError:
        raise ValueError('Mask is empty. Cannot compute a bounding box!')

    return tuple(reversed(out))

def update_pos_with_bb(pos,bb):
    return np.add(pos, [b.start for b in bb])

def pad_array_centered(array, position ,max_shape):
    # Calculate the required padding in height, width, and depth directions
    pad_height = max_shape[0] - array.shape[0]
    pad_width = max_shape[1] - array.shape[1]
    pad_depth = max_shape[2] - array.shape[2]

    # Calculate the padding for each side of the array along each axis
    pad_front = pad_height // 2
    pad_back = pad_height - pad_front
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_depth // 2
    pad_bottom = pad_depth - pad_top

    # Create a new array with the required padding on each side
    padded_array = np.pad(array, ((pad_front, pad_back), (pad_left, pad_right), (pad_top, pad_bottom)), mode='constant', constant_values=0)
    position_update = np.add(position,(-pad_front, -pad_left, -pad_top))
    
    return padded_array, position_update

def pad_and_crop_position(array, position, new_position):
    # Calculate the required padding in x, y, and z directions
    pad_x = max(position[0] - new_position[0], 0)
    pad_y = max(position[1] - new_position[1], 0)
    pad_z = max(position[2] - new_position[2], 0)

    # Calculate the required cropping in x, y, and z directions
    crop_x = max(new_position[0] - position[0], 0)
    crop_y = max(new_position[1] - position[1], 0)
    crop_z = max(new_position[2] - position[2], 0)

    # Create a new array with the required padding
    padded_array = np.pad(array, ((pad_x, 0), (pad_y, 0), (pad_z, 0)), mode='constant', constant_values=0)

    # Calculate the shape of the cropped array
    cropped_shape = (
        padded_array.shape[0] - crop_x,
        padded_array.shape[1] - crop_y,
        padded_array.shape[2] - crop_z
    )

    # Crop the array if necessary
    crop_x = np.max([crop_x,0])
    crop_y = np.max([crop_y,0])
    crop_z = np.max([crop_z,0])
    
    cropped_array = padded_array[crop_x:, crop_y:, crop_z:]


    return cropped_array


