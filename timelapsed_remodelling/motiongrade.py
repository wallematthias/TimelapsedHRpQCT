
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from aim import aim
from skimage.filters import gaussian
from skimage.measure import label   
from glob import glob
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from IPython.display import display, clear_output 
from IPython.display import Image as iImage



def pad_to_square(image):
    height, width = image.shape[:2]
    size = max(height, width)
    top_pad = (size - height) // 2
    bottom_pad = size - height - top_pad
    left_pad = (size - width) // 2
    right_pad = size - width - left_pad
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    return padded_image

def preprocess_image(image):
    # Pad the image to square dimensions
    padded_image = pad_to_square(image)

    # Normalize and convert to CV_8U dtype
    normalized_image = cv2.normalize(padded_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to PIL Image
    pil_image = Image.fromarray(normalized_image)

    # Resize the image to [512, 512]
    resized_image = pil_image.resize((512, 512))

    return img_to_array(resized_image)

def plot_slice(data, slices, scores):
    n_slices = len(slices) + 1
    # Compute the optimal figure width based on the size of the data array
    data_shape = np.rot90(data[:, :, slices[0]]).shape
    fsize = 3
    fig_width = max(1, n_slices * (data_shape[1] / data_shape[0]) * fsize)
    fig, axes = plt.subplots(1, n_slices, figsize=(fig_width, fsize), facecolor='white')
    for ax, s, score in zip(axes.flatten()[::-1], slices, scores):
        img = ax.imshow(np.rot90(data[:, :, s]))
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('# {} (grade {})'.format(s, score), fontsize=10)
    fig.subplots_adjust(wspace=0.05)
    return fig, axes

def get_indices(scores, confidences):
    score_dict = {}
    for i, score in enumerate(scores):
        if score not in score_dict:
            score_dict[score] = (i, score, confidences[i])
        else:
            if confidences[i] > score_dict[score][2]:
                score_dict[score] = (i, score, confidences[i])

    sorted_scores = sorted(score_dict.keys())

    lowest = sorted_scores[0]
    highest = sorted_scores[-1]
    if len(sorted_scores) % 2 == 0:
        median = sorted_scores[len(sorted_scores) // 2 - 1]
    else:
        median = sorted_scores[len(sorted_scores) // 2]

    indices = [score_dict[lowest][0], score_dict[median][0], score_dict[highest][0]]
    scores = [score_dict[lowest][1], score_dict[median][1], score_dict[highest][1]]

    if len(sorted_scores) < 3:
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        indices = [sorted_indices[0], sorted_indices[len(sorted_indices) // 2], sorted_indices[-1]]
        scores = [scores[sorted_indices[0]], scores[len(sorted_indices) // 2], scores[sorted_indices[-1]]]

    return indices, scores

def automatic_motion_score(im_raw, outpath=None, stackheight=168):
    
    if stackheight>im_raw.shape[2]:
        print(f'Stackheight input {stackheight} larger than image {im_raw.shape[2]} reducing stackheight')
        stackheight = im_raw.shape[2]

    model_paths = sorted(glob(os.path.join(
            os.path.dirname(
            Path(__file__)),
            'models','*.h5')))
    DNN_list = [keras.models.load_model(model_path, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()}) for model_path in model_paths]

    for i, DNN_model in enumerate(DNN_list):
        DNN_model._name = "model" + str(i)

    im_filtered = gaussian(im_raw, sigma=0.8, truncate=1.25)

    model_number = 10


    full_scan = np.asarray([preprocess_image(im_filtered[:, :, i]) for i in range(0, im_filtered.shape[2])])
    

    result = np.zeros((len(DNN_list), full_scan.shape[0], 5))
    for i in range(len(DNN_list)):
        result[i] = DNN_list[i].predict(full_scan / 255)

    all_result_binary = np.zeros(result.shape)
    all_result_matrix = np.zeros(result.shape[:2])

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            max_ = max(result[i, j])
            all_result_binary[i, j] = np.floor(result[i, j] / max_)
            all_result_matrix[i, j] = np.where(all_result_binary[i, j] == 1)[0][0]

    votes = []
    for i in range(all_result_matrix.shape[1]):
        bins = np.bincount(all_result_matrix[:, i].astype('int8'), minlength=5)
        votes.append(bins)
    votes = np.array(votes) / model_number

    category_colors_neg = plt.get_cmap('RdBu')(np.linspace(0.75, 0.95, 3))
    category_colors_pos = plt.get_cmap('RdBu')(np.linspace(0.05, 0.15, 2))

    class_1 = votes[:, 0]
    class_2 = votes[:, 1]
    class_3 = votes[:, 2]
    class_4 = votes[:, 3]
    class_5 = votes[:, 4]

    mscore = np.round(np.mean(np.argmax(votes, axis=1) + 1), 0)
    mscorevalue = np.round(np.mean(np.max(votes, axis=1)), 2)
    score = np.round(np.mean(np.reshape(np.argmax(votes, axis=1) + 1, (int(full_scan.shape[0] / stackheight), stackheight)), axis=1), 0)
    scorevalue = np.round(np.mean(np.reshape(np.max(votes, axis=1), (int(full_scan.shape[0] / stackheight), stackheight)), axis=1), 2)

    slice_score = np.argmax(votes, axis=1) + 1
    slice_conf = np.max(votes, axis=1)

    data = im_raw
    slices, scores = get_indices(slice_score, slice_conf)
    fig, axes = plot_slice(data, slices, scores)
    ax = axes[0]

    labels = range(0, full_scan.shape[0])

    ax.barh(y=labels, width=class_1, height=1, left=-(class_1 + class_2 + class_3), label='score 1', color=category_colors_neg[2])
    ax.barh(y=labels, width=class_2, height=1, left=-np.add(class_2, class_3), label='score 2', color=category_colors_neg[1])
    ax.barh(y=labels, width=class_3, height=1, left=-class_3, label='score 3', color=category_colors_neg[0])
    ax.barh(y=labels, width=class_4, height=1, label='score 4', color=category_colors_pos[1])
    ax.barh(y=labels, width=class_5, height=1, left=class_4, label='score 5', color=category_colors_pos[0])

    fig.legend(fontsize=10, loc='lower center', ncol=5)

    ax.set_ylim(0, full_scan.shape[0])
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Slice #', fontsize=10, labelpad=10)
    ax.set_xlabel('')
    fig.suptitle('{} stacks ({} slices) Score {} ({})'.format(int(full_scan.shape[0] / stackheight), stackheight, mscore, mscorevalue), fontsize=12, y=0.95)
    ax.grid('on', axis='y', linewidth=0.5)

    fig.subplots_adjust(bottom=0.2)

    if outpath is None:
        outpath = os.path.dirname(path)
    else:
        outpath = os.path.abspath(outpath)

    for i, (y, s, val) in enumerate(zip(np.linspace(0, full_scan.shape[0] - stackheight, (int(full_scan.shape[0] / stackheight))), score, scorevalue)):
        if i > 0:
            ax.axhline(y, linestyle='--')
        t = ax.text(-0.9, y + stackheight / 2, 'Score {} ({})'.format(s, val), fontsize=8)
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))

    # Convert mscorevalue to a percentage
    mscore_percentage = int(mscorevalue * 100)
    mscore_int = int(mscore)

    # Create the filename using f-string formatting
    filename = f'{outpath}_{mscore_int}_{mscore_percentage}_motion.png'
        
    plt.savefig(filename, transparent=False)
    plt.show()
    plt.close(fig)
    if len(score) > 1:
        return np.append(score, mscore), np.append(scorevalue, mscorevalue)
    else:
        return mscore, mscorevalue

def grade_images(image_folder_path,stackheight,outpath):

    paths = glob(image_folder_path)

    # Keywords to exclude (case-insensitive)
    exclude_keywords = ['mask', 'trab', 'cort']
    
    # Filter out images containing excluded keywords in their filename
    paths = [path for path in paths if not any(keyword.lower() in os.path.basename(path).lower() for keyword in exclude_keywords)]
    
    for path in paths:
        file = aim.load_aim(path)
        name = os.path.basename(path).split('.')[0]
        mscore, mscorevalue = automatic_motion_score(
            file.data, outpath=os.path.join(outpath,name), stackheight=stackheight)
        
        print('Motion Score {}: {}'.format(name,mscore))
    
def confirm_images(image_folder_path, confidence_threshold, output_path):
    image_files = glob(image_folder_path)

    data = []
    total_images = len(image_files)
    graded_images = 0

    for image_path in image_files:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        file_parts = file_name.split('_')

        if len(file_parts) >= 3:
            motion_score_default = int(file_parts[-3])  
            confidence_default = int(file_parts[-2])
            filename = '_'.join(file_parts[:-3]) 
        else:
            motion_score_default = ""
            confidence_default = ""
            filename = ""

        grading_type = "automatic" if confidence_default >= confidence_threshold else "manual"

        if confidence_default < confidence_threshold:
            clear_output(wait=True)
            graded_images += 1
            remaining_images = total_images - graded_images
            print(f"Graded {graded_images} out of {total_images} images. {remaining_images} images remaining.")
            display(iImage(filename=image_path))
            motion_score = input(f"Enter your assessment for the motion score [{motion_score_default}]: ")
            if not motion_score:
                motion_score = motion_score_default

        else:
            motion_score = motion_score_default
            graded_images += 1

        data.append({'filename': filename, 'manual_grade': int(motion_score), 'automatic_grade': int(motion_score_default), 'confidence': int(confidence_default)})

    data_df = pd.DataFrame(data)
    data_df.to_csv(output_path, index=False)

    correct_predictions = (data_df['manual_grade'] == data_df['automatic_grade']).sum()
    total_predictions = len(data_df)
    accuracy = correct_predictions / total_predictions

    print("Grading completed. Graded data saved to '{}'.".format(output_path))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
def main(option='grade'):
    # Get user input or specify the arguments here
    
    
    if option == 'confirm':
        image_folder_path = input("Enter the image folder path (e.g., /path/to/data/*motion.png): ")
        confidence_threshold = int(input("Enter the confidence threshold (e.g., 75 [%]): "))
        output_path = input("Enter the output file path for graded data (e.g., graded_data.csv): ")
    
        # Call the function with user-provided arguments
        confirm_images(image_folder_path, confidence_threshold, output_path)
        
    elif option=='grade':
        image_folder_path = input("Enter the image folder path (e.g. /path/to/data/*.AIM): ")
        stackheight = int(input("Enter the stackheight (e.g. 168): "))
        output_path = input("Enter the output file path for graded data (e.g., /path/to/output/data/): ")
        
        grade_images(image_folder_path,stackheight,output_path)
    else: 
        print('Enter valid option grade/confirm')
    
    
if __name__ == "__main__":
    main()