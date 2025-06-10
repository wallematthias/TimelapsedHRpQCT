import pandas as pd
import numpy as np
import statsmodels.api as sm

from multiprocessing import Pool
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
import SimpleITK as sitk
import argparse
import os 
from timelapsed_remodelling.reposition import pad_array_centered
import os 
from glob import glob
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def overlay_plot(ax, finite_element_image, masks):
    # Create a colormap for the finite element image (grayscale)
    cmap_gray = plt.cm.coolwarm
    
    # Create a colormap for the masks
    cmap_masks = ListedColormap(['yellow', 'orange', 'purple', 'white'])  # Add your desired colors for resorption and ROI
    
    # Plot the finite element image in grayscale
    ax.imshow(finite_element_image, cmap=cmap_gray)

    # Plot each mask with its corresponding color
    for i, mask in enumerate(masks):
        ax.contour(mask, levels=[0.5], colors=[cmap_masks(i)])

    # Customize the plot
    ax.axis('off')  # Turn off axis labels and ticks

def confusion_matrix_per_threshold_set(data_per_category, threshold_list):
    """
    Function to compute the confusion matrix (True vs Predicted data) for a set of data spanning the same independent variable (Ex: Formation, Quiescence, Resorption histogram counts across normalised SED bins).
    
    The input data must be ordered in the expected order of the thresholds that will separate each pair of categories
    Ex: For SED mechanoregulation analysis with Schulte's method, conditional probability curves will appear as R -> Q -> F, so the data_per_category list should be sorted as [[R_data], [Q_data], [F_data]]
    
    Parameters
    ----------
    data_per_category : list of lists/arrays
        List containing one list/array per category of the data being analysed (ex: (re)modelling events).
        
    threshold_list : list of ints
        List containing the indices determining the 'True data' thresholds where the data should be split.
    
    Returns
    -------
    confusion_matrix : np.ndarray
        2D numpy array with the confusion matrix of the set of thresholds given as input.

    """

    size_first_cat_data = len(data_per_category[0])
    list_sizes = []
    for list_idx, data_list in enumerate(data_per_category):
        list_sizes.append(len(data_list))
    
    if np.unique(np.array(list_sizes)).size != 1:
        raise ValueError(f"The data lists passed as input have different sizes: {list_sizes}.")
    
    size_conf_matrix = len(data_per_category)
    confusion_matrix = np.zeros((size_conf_matrix, size_conf_matrix))
                
    for row, category in enumerate(data_per_category):
        for col, sl in enumerate(slice(a,b) for a,b in zip([0,] + threshold_list, threshold_list + [len(category),])):
            confusion_matrix[row, col] = np.sum(category[sl])
    
    confusion_matrix /= np.sum(confusion_matrix)
    
    return confusion_matrix

def ccr_from_conf_matrix(data_per_category, threshold_list):
    """
    Function to compute the Correct Classification Coefficient of a confusion matrix.
    
    The input data must be ordered in the expected order of the thresholds that will separate each pair of categories
    Ex: For SED mechanoregulation analysis with Schulte's method, conditional probability curves will appear as R -> Q -> F, so the hist_data list should be sorted as [[R_data], [Q_data], [F_data]]

    Parameters
    ----------
    data_per_category : list of lists/arrays
        List containing one list/array per category of the data being analysed (ex: (re)modelling events).
        
    threshold_list : list of ints
        List containing the indices determining the 'True data' thresholds where the categorical data should be split.
    
    Returns
    -------
    ccr : float
        Value of the Correct Classification Rate for the set of thresholds given as input.

    """
    
    size_first_cat_data = len(data_per_category[0])
    list_sizes = []
    for list_idx, data_list in enumerate(data_per_category):
        list_sizes.append(len(data_list))
    
    if np.unique(np.array(list_sizes)).size != 1:
        raise ValueError(f"The data lists passed as input have different sizes: {list_sizes}.")

    conf_matrix = confusion_matrix_per_threshold_set(data_per_category, threshold_list)
    ccr = np.matrix.trace(conf_matrix)

    return ccr

def max_ccr(hist_data, hist_bin_edges):
    """
    Function to compute the maximum Correct Classification Coefficient of a set of histogram data.
    
    The input data must be ordered in the expected order of the thresholds that will separate each pair of categories
    Ex: For SED mechanoregulation analysis with Schulte's method, conditional probability curves will appear as R -> Q -> F, so the hist_data list should be sorted as [[R_data], [Q_data], [F_data]]

    Parameters
    ----------
    hist_data : list of lists/arrays
        List containing one list/array of histogram counts per category of the data being analysed (ex: (re)modelling events).
        
    hist_bin_edges : np.ndarray
        Array containing the bin edges used to compute the histograms given as input.
    
    Returns
    -------
    full_ccr_matrix : np.ndarray
        Matrix containing the CCR value for all set of thresholds possible based on the bin edges used to compute the histograms.
        
    max_point_coords : np.ndarray
        Coordinates of the maximum CCR value in the matrix 'current_ccr_matrix'

    """
    
    size_first_cat_data = len(hist_data[0])
    list_sizes = []
    for list_idx, data_list in enumerate(hist_data):
        list_sizes.append(len(data_list))
    
    if np.unique(np.array(list_sizes)).size != 1:
        raise ValueError(f"The data lists passed as input have different sizes: {list_sizes}.")
    
    num_thresholds = len(hist_data) - 1
    thresh_list = np.squeeze((hist_bin_edges[1:] + hist_bin_edges[:-1]) / 2)
    
    full_ccr_matrix = np.full(np.repeat(thresh_list.size - 1, num_thresholds), np.nan)

    thresh_idx_list = np.arange(0, thresh_list.size-1)
    full_thresh_set = combinations(thresh_idx_list, num_thresholds)
    
    for thresh_idx, thresh_set in enumerate(full_thresh_set):
        full_ccr_matrix[tuple(thresh_set)] = ccr_from_conf_matrix(hist_data, list(thresh_set))

    max_point_coords = np.where(full_ccr_matrix == np.nanmax(full_ccr_matrix))

    return full_ccr_matrix, max_point_coords

def remXmech_regression(event,strains,support):
    y = np.asarray(event)
    X = np.c_[np.ones_like(strains), np.asarray(strains)]

    #logit = sm.GLM(y, X, family=sm.families.Binomial()).fit() # notice the order of the endog and exog variables
    logit = sm.Logit(y, X,).fit(disp=0) # notice the order of the endog and exog variables
    #print(logit.summary())
    # New data for the prediction
    xnew = np.c_[np.ones(support.size), support]  # must be a 2D array
    out_sm = logit.predict(xnew)

    return out_sm, logit

def mechreg_regression(R_strains,Q_strains,F_strains):
    
    F_class = [1,]*len(F_strains) + [0,]*len(R_strains) + [0,]*len(Q_strains)
    strain = list(F_strains)+ list(R_strains) +list(Q_strains)
    R_class = [0,]*len(F_strains) + [1,]*len(R_strains) + [0,]*len(Q_strains)
    
    support = np.linspace(0, np.max(strain), 100)
    
    y_pred_form, model_form = remXmech_regression(F_class,strain,support)
    y_pred_res, model_res = remXmech_regression(R_class,strain,support)
    y_pred_qui =  1-(y_pred_form+y_pred_res)
    
    return  y_pred_res, y_pred_qui, y_pred_form, support, model_res, model_form

def ci(var,alpha=0.01):
    coef = np.asarray(var)
    c_mean = np.mean(coef)
    c_std = np.std(coef)
    ql = (alpha/2)*100.
    qh = (1 - alpha/2)*100.
    ci_low = np.percentile(coef, ql, method='midpoint')
    ci_high = np.percentile(coef, qh, method='midpoint')
    return c_mean, c_std, ci_low, ci_high

def parMechreg(arr):
    
        R, Q, F = arr
       
        r, q, f, bins, model_res, model_form = mechreg_regression(R,Q,F)
        
        n_R,_ =np.histogram(R, bins=bins, density=True)
        n_Q,_ =np.histogram(Q, bins=bins, density=True)
        n_F,_ =np.histogram(F, bins=bins, density=True)

        n_total = np.sum([n_Q,n_R,n_F],axis=0)

        p_Q = np.divide(n_Q,n_total, out=np.zeros_like(n_Q), where=n_total!=0)
        p_F = np.divide(n_F,n_total, out=np.zeros_like(n_F), where=n_total!=0)
        p_R = np.divide(n_R,n_total, out=np.zeros_like(n_R), where=n_total!=0)

        data = {}
        data['r'] = r
        data['q'] = q
        data['f'] = f 
        data['bins'] = bins
        
        data['const_res'] = model_res.params[0] # constant term
        data['x1_res']=model_res.params[1]   # x1 value
        data['p_const_res']=model_res.pvalues[0]
        data['p_x1_res']=model_res.pvalues[1]
        data['const_form']=model_form.params[0] # constant term
        data['x1_form']=model_form.params[1]    # x1 value
        data['p_const_form']=model_form.pvalues[0] # p const
        data['p_x1_form'] = model_form.pvalues[1]  # p x1
        try: 
            ccr_matrix, max_coords = max_ccr([p_R, p_Q, p_F], bins)
            data['ccr'] = ccr_matrix[max_coords][0]
            data['thr_res'] = max_coords[0][0]
            data['thr_form'] = max_coords[1][0]
        except TypeError:
            data['ccr'] = ccr_matrix[max_coords]
            data['thr_res'] = max_coords[0]
            data['thr_form'] = max_coords[1]        
        except:
            data['ccr'] = 0
            data['thr_res'] = 0
            data['thr_form'] = 0                 

        return data
        
def bootstrap_mechreg(R_strains,Q_strains,F_strains,n_boot=10,alpha=0.01,perc_surf=10):

    values = []
    min_bin = int(np.round(np.min([len(R_strains),len(Q_strains),len(F_strains)])/100*perc_surf))
    print('Using k={} sampling {} values'.format(n_boot, min_bin))
    R_samples = np.random.choice(R_strains,[n_boot, min_bin],replace=True)
    Q_samples = np.random.choice(Q_strains,[n_boot, min_bin],replace=True)
    F_samples = np.random.choice(F_strains,[n_boot, min_bin],replace=True)

    res = []; qui = []; form = []
    
    const_res = []; x1_res = []; p_const_res = []; p_x1_res = []
    const_form = []; x1_form = []; p_const_form = []; p_x1_form = []
    ccr = []; thr_res = []; thr_form = []
    
    arrays = zip(R_samples, Q_samples, F_samples)
    
    with Pool() as pool:
        # call the same function with different data in parallel
        for data in pool.map(parMechreg, arrays):
            
            ccr.append(data['ccr'])
            thr_res.append(data['thr_res'])
            thr_form.append(data['thr_form'])

            res.append(data['r'])
            qui.append(data['q'])
            form.append(data['f'])

            const_res.append(data['const_res'])  # constant term
            x1_res.append(data['x1_res'])     # x1 value
            p_const_res.append(data['p_const_res']) 
            p_x1_res.append(data['p_x1_res']) 
            const_form.append(data['const_form'])  # constant term
            x1_form.append(data['x1_form'])     # x1 value
            p_const_form.append(data['p_const_form']) # p const
            p_x1_form.append(data['p_x1_form'])   # p x1
            bins = data['bins']
    
    ql = (alpha/2)*100.
    qh = (1 - alpha/2)*100.

    means_res = np.mean(res, axis=0)
    stds_res = np.std(res, axis=0)
    ci_lows_res = np.percentile(res, ql, axis=0, method='midpoint')
    ci_higs_res = np.percentile(res, qh, axis=0, method='midpoint')
    
    x1m_res, x1s_res, x1l_res, x1h_res = ci(x1_res,alpha) 

    means_form = np.mean(form, axis=0)
    stds_form = np.std(form, axis=0)
    ci_lows_form = np.percentile(form, ql, axis=0, method='midpoint')
    ci_higs_form = np.percentile(form, qh, axis=0, method='midpoint')
    
    x1m_form, x1s_form, x1l_form, x1h_form = ci(x1_form,alpha) 
    
    means_qui = np.mean(qui, axis=0)
    stds_qui = np.std(qui, axis=0)
    ci_lows_qui = np.percentile(qui, ql, axis=0, method='midpoint')
    ci_higs_qui = np.percentile(qui, qh, axis=0, method='midpoint')
    
    form_curves = {}
    form_curves['x_bins'] = bins
    form_curves['y_means'] = means_form
    form_curves['y_low'] = ci_lows_form
    form_curves['y_high'] = ci_higs_form
    form_metrics = {}
    
    form_metrics['pvalue'] = np.mean(p_x1_form)
    form_metrics['odds'] = np.exp(x1m_form)
    form_metrics['low_odds'] = np.exp(x1l_form)
    form_metrics['high_odds'] = np.exp(x1h_form)

    res_curves = {}
    res_curves['x_bins'] = bins
    res_curves['y_means'] = means_res
    res_curves['y_low'] = ci_lows_res
    res_curves['y_high'] = ci_higs_res
    
    res_metrics = {}
    res_metrics['pvalue'] = np.mean(p_x1_res)
    res_metrics['odds'] = np.exp(x1m_res)
    res_metrics['low_odds'] = np.exp(x1l_res)
    res_metrics['high_odds'] = np.exp(x1h_res)
    
    qui_curves = {}
    qui_curves['x_bins'] = bins
    qui_curves['y_means'] = means_qui
    qui_curves['y_low'] = ci_lows_qui
    qui_curves['y_high'] = ci_higs_qui
 
    print(ccr)
    ccrm, ccrs, ccrl, ccrh = ci(ccr,alpha) 
    thr_res_m, thr_res_s, thr_res_l, thr_res_h = ci(thr_res,alpha) 
    thr_form_m, thr_form_s, thr_form_l, thr_form_h = ci(thr_form,alpha) 

    glob_metrics = {}
    glob_metrics['CCR_mean'] = ccrm
    glob_metrics['CCR_low'] = ccrl
    glob_metrics['CCR_high'] = ccrh
    glob_metrics['thr_form_mean'] = thr_form_m
    glob_metrics['thr_form_low'] =thr_form_l
    glob_metrics['thr_form_high'] = thr_form_h
    glob_metrics['thr_res_mean'] = thr_res_m
    glob_metrics['thr_res_low'] = thr_res_l
    glob_metrics['thr_res_high'] = thr_res_h
    
    return res_metrics, form_metrics, glob_metrics, res_curves, qui_curves, form_curves

def remodelling_x_strain(remodelling_image,baseline_strain,mask=None, figname=None):

    if mask is None:
        mask = np.ones_like(remodelling_image)>0    
    
    eval_mask = (mask>0) & (remodelling_image>0) #this should only include the common region

    # We norm this
    baseline_strain[eval_mask==0]=0
    perc99 = np.percentile(baseline_strain[eval_mask],99)
    baseline_strain[baseline_strain>perc99]=perc99

    bin_baseline = baseline_strain>0 #strain is 0 in the background
    bone_surface = (mask>0) & np.bitwise_xor(bin_baseline, binary_erosion(bin_baseline))
    resorption = (mask>0) & (remodelling_image == 1)
    formation = (mask>0) & (remodelling_image == 3)
    dilated_resorption = (mask>0) & binary_dilation(resorption, iterations=1) #1-2
    dilated_formation = (mask>0) & binary_dilation(formation, iterations=1) #1-2

    surface_formation = dilated_formation[bone_surface]
    surface_resorption = dilated_resorption[bone_surface]
    surface_signal = baseline_strain[bone_surface]

    data = {'F':surface_formation, 'R': surface_resorption, 'MS':surface_signal}
    
    if figname is not None:
        fig, ax = plt.subplots()
        x = 100
        overlay_plot(ax, baseline_strain[:,:,x], [bone_surface[:,:,x],dilated_formation[:,:,x],dilated_resorption[:,:,x],mask[:,:,x]])
        fig.savefig(figname)
    
    # Return everything in a dataframe 
    return pd.DataFrame(data)

def mechanoregulation(remodelling_image,baseline_strain,mask=None,nbins=100,alpha=0.01,n_boot=1000,sampling_perc=10, meta_data=None):

    if meta_data is None:
        meta_data = {}

    # Add some meta data 
    meta_data['alpha'] = alpha
    meta_data['sampling_perc'] = sampling_perc
    meta_data['n_boot'] = n_boot
    
    if meta_data['figname'] is not None:
        figname=meta_data['figname']
    else:
        figname=None
        
    raw_df = remodelling_x_strain(remodelling_image,baseline_strain,mask=mask, figname=figname)

    # Set 'MS' values greater than 1 to 1
    raw_df.loc[raw_df['MS'] > 1, 'MS'] = 1

    # Create a 'fqr' array with 'Q', 'R', and 'F' values
    fqr = np.full(len(raw_df), 'Q')
    fqr[raw_df['F'].values] = 'F'
    fqr[raw_df['R'].values] = 'R'

    # Use boolean indexing to get subsets
    Q = raw_df.loc[fqr == 'Q', 'MS']
    R = raw_df.loc[fqr == 'R', 'MS']
    F = raw_df.loc[fqr == 'F', 'MS']

    # Calculate mechanoregulation metrics
    res_metrics, form_metrics, glob_metrics, res_curves, qui_curves, form_curves = bootstrap_mechreg(R,Q,F,n_boot,alpha,sampling_perc)
    
    # Form dataframes
    solution_df = pd.DataFrame(res_metrics,index=[0]).join(pd.DataFrame(form_metrics,index=[0]),lsuffix='_res', rsuffix='_form').join(pd.DataFrame(glob_metrics,index=[0]))
    form_df = pd.DataFrame(form_curves)
    res_df = pd.DataFrame(res_curves)
    qui_df = pd.DataFrame(qui_curves)

    # List of dataframes
    dataframes = [solution_df, res_df, form_df, qui_df, raw_df]

    # Adding each meta data dictionary entry as a new column to each dataframe
    for df in dataframes:
        for key, value in meta_data.items():
            df[key] = value

    [solution_df, res_df, form_df, qui_df, raw_df] = dataframes

    return solution_df, res_df, form_df, qui_df, raw_df

def main(args):
    try:
        rem_path = args['rem_path']

        
        lookup = {0:'00',1:'06',2:'12',3:'18'}

        b = int(rem_path.replace('.','_').split('_')[-3])
        f = int(rem_path.replace('.','_').split('_')[-2])
        site = os.path.basename(rem_path).split('_')[2]
        id = os.path.basename(rem_path).split('_')[1]
        participant_pre = '_'.join(os.path.basename(rem_path).split('_')[:3])
        participant = '_'.join(os.path.basename(rem_path).split('_')[:4])

        sed_path = os.path.join(os.path.foldername(rem_path),f'{participant_pre}_M{lookup[b]}_HOM_LS_sed.mha')

        print(rem_path)
        print(sed_path)


        meta_data = {}
        meta_data['patient'] = participant
        meta_data['site'] = site
        meta_data['id'] = id
        meta_data['baseline'] = b
        meta_data['followup'] = f

        
        if args['mask_type'] is not None:
            mask_type = args['mask_type']
            meta_data['mask'] = mask_type
            trab_path = rem_path.replace('remodelling', mask_type)
            print(trab_path)
            trab = sitk.ReadImage(trab_path)
            trab.SetSpacing([1, 1, 1])
            trab = np.swapaxes(sitk.GetArrayFromImage(trab), 0, 2)
        else:
            trab = None
            meta_data['mask'] = 'FULLMASK'

        maskname = f"{meta_data['mask']}_mechreg"
        meta_data['figname'] = rem_path.replace('remodelling', maskname).replace('.mha', '.png')
        
        rem_image = sitk.ReadImage(rem_path)
        sed_image = sitk.ReadImage(sed_path)

        rem_image.SetSpacing([1, 1, 1])
        sed_image.SetSpacing([1, 1, 1])

        remodelling_image = np.swapaxes(sitk.GetArrayFromImage(rem_image), 0, 2)
        baseline_strain = np.swapaxes(sitk.GetArrayFromImage(sed_image), 0, 2)[:, :, 0:167]
        baseline_strain, new_pos = pad_array_centered(baseline_strain, sed_image.GetOrigin()[::-1], remodelling_image.shape)

        solution_df, res_df, form_df, qui_df, raw_df = mechanoregulation(remodelling_image, baseline_strain, mask=trab,
                                                                          nbins=100, alpha=0.01, n_boot=100,
                                                                          sampling_perc=100, meta_data=meta_data)

        solution_df.to_csv(rem_path.replace('remodelling', maskname).replace('.mha', '.csv'))
        print(solution_df)

        base_approx = (remodelling_image < 3) & (remodelling_image > 0)
        overlay = (baseline_strain > 0).astype(float) * 2 + base_approx.astype(float)


    except Exception as e:
        # Handle the exception here
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("rem_path", help="Path to remodelling file")
    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")
    parser.add_argument("--mask_type", default=None, help="Type of mask")

    args = parser.parse_args()

    if args.multiprocessing:
        # Get the list of AIM files
        files = glob(f'{args.rem_path}/**/*remodelling*.mha')

        for file in files:
            main({'rem_path': file, 'mask_type': args.mask_type})
    else:
        main({'rem_path': args.rem_path, 'mask_type': args.mask_type})
