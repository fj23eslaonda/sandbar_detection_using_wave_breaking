"""
PURPOSE  : Identify Sand Bar using wave breaking patterns.
AUTHOR   : Francisco Sáez R.
EMAIL    : francisco.saez@sansano.usm.cl
Date     : 01/12/2021

This code allow identifying sand bars position through a method
propose by Sáez et al. (2021)

"""

# -----------------------------------------------------------------
# 
# PACKAGES
#
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import find_peaks, detrend
import pandas as pd
import cv2


# -----------------------------------------------------------------
#
# CONCATENATE
#
# -----------------------------------------------------------------
#
def plot_img_and_mask(original_img, mean_mask, main_path, beach_path, x_point, y_point, orientation):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    plt.suptitle('The Sand Bar detection result on '+ str(beach_path[1:-1]), fontsize=20)

    axs[0].imshow(original_img / np.max(original_img))
    map1 = axs[0].imshow(mean_mask, vmax=np.nanmax(mean_mask), cmap='turbo')
    if orientation == 'vertical':
        axs[0].set_xlabel('Alongshore distance, y [pixels]', fontsize=14)
        axs[0].set_ylabel('Cross-shore distance, x [pixels]', fontsize=14)
    else:
        axs[0].set_ylabel('Alongshore distance, y [pixels]', fontsize=14)
        axs[0].set_xlabel('Cross-shore distance, x [pixels]', fontsize=14)
    axs[0].set_title('Cumulative Breaking Map', fontsize=14)
    axs[0].grid(linestyle='--', alpha=0.7)
    fig.colorbar(map1, ax=axs[0], fraction=0.05, pad=0.04)

    axs[1].imshow(original_img / np.max(original_img), vmax=np.max(original_img))
    if orientation == 'vertical':
        axs[1].scatter(x_point, y_point, marker='x', c='r', s=8, label='Wave Breaking Method')
        axs[1].set_xlabel('Alongshore distance, y [pixels]', fontsize=14)
    else:
        axs[1].scatter(y_point, x_point, marker='x', c='r', s=8, label='Wave Breaking Method')
        axs[1].set_xlabel('Cross-shore distance, x [pixels]', fontsize=14)
    
    axs[1].set_title('Average Image', fontsize=14)
    axs[1].grid(linestyle='--', alpha=0.7)
    axs[1].legend()

    saving_path = main_path + beach_path + '/sandbar_results/'
    plt.savefig(saving_path + '/cumulative_breaking.png',
                bbox_inches='tight',
                pad_inches=1)
    plt.show()
    plt.close("all")

    # SAVE MATRIX
    np.save(saving_path + '/mean_image', original_img)
    np.save(saving_path + '/mean_mask', mean_mask)


# -----------------------------------------------------------------
#
# INTERPOLATE CUMULATIVE BREAKING PIXELS MAP
#
# -----------------------------------------------------------------
def interp_cumulative_breaking(mean_mask, size):
    """
    Returns interpolated cumulative breaking maps
    :param size: size of matrix
    :param mean_mask: cumulative breaking pixels map
    :return: inter_mean_mask
    """
    mean_mask = delete_edges(mean_mask, 12)
    mean_mask = cv2.resize(mean_mask * 255.0,
                           (size['old_width'], size['old_height']),
                           interpolation=cv2.INTER_AREA) / 255.0
    # ZEROS TO NAN
    mean_mask[mean_mask == 0] = np.nan

    # NO NAN INDICES
    ix_no_nan = ~ np.isnan(mean_mask)

    # GRID
    shape = np.shape(mean_mask)
    x = np.arange(1, shape[1] + 0.1, 1)
    y = np.arange(1, shape[0] + 0.1, 1)
    X, Y = np.meshgrid(x, y)

    # INTERPOLATION
    points = np.dstack((X[ix_no_nan], Y[ix_no_nan]))[0, :, :]
    values = mean_mask[ix_no_nan]
    inter_mean_mask = griddata(points, values, (X, Y), method='linear')

    return inter_mean_mask


# -----------------------------------------------------------------
#
# IDENTIFY SAND BAR POINTS
#
# -----------------------------------------------------------------
def identify_sandbar_pts(mean_mask, orientation):
    """
    Returns two lists with all sandbar points
    :param orientation: Waves direction
    :param mean_mask: cumulative breaking pixels map
    :return: x_point and y_point
    """
    mean_mask = np.nan_to_num(mean_mask)
    x_point, y_point = list(), list()
    for i in range(len(mean_mask[0, :])):
        if orientation == 'vertical':
            row = mean_mask[:, i]
        else:
            row = mean_mask[i, :]
        peaks, _ = find_peaks(detrend(smooth_hamming(row, 100)), prominence=4)
        for peak in peaks:
            x_point.append(i)
            y_point.append(peak)

    mean_mask[mean_mask <= np.nanmin(mean_mask)] = np.nan

    return x_point, y_point, mean_mask


# -----------------------------------------------------------------
#
# FILTER TO
#
# -----------------------------------------------------------------
def smooth_hamming(ts, window):
    """
    Hamming smooth function using pandas
    :param ts: 1D np array
    :param window: number of points for hamming filter
    :return: smooth_ts
    """
    df = pd.DataFrame(columns=['ts'])
    df.ts = ts
    df['smooth_ts'] = df.iloc[:, 0].rolling(window,
                                            min_periods=1,
                                            win_type='hamming',
                                            center=True).mean()
    smooth_ts = np.squeeze(np.array(df.smooth_ts))
    return smooth_ts


# -----------------------------------------------------------------
#
# DELETE EDGES
#
# -----------------------------------------------------------------
def delete_edges(mask, delta):
    """
    Returns mask without edge and this space will be filled with
    interp_cumulative_breaking function
    :param delta: number of pixels to be deleted
    :param mask: mask predicted
    :return: mask without edge
    """
    shape = np.shape(mask)
    # X DIRECTION
    for ix in range(shape[1] // 512 - 1):
        mask[:, 512 * (ix + 1) - delta:512 * (ix + 1) + delta] = 0
    # Y DIRECTION
    for ix in range(shape[0] // 512 - 1):
        mask[512 * (ix + 1) - delta:512 * (ix + 1) + delta, :] = 0
    return mask


# -------------------------------------------------------------------------------------
#
# LOAD DATA
#
# -------------------------------------------------------------------------------------
def plot_predictions(image, mask, beach_path, name, size):
    """
    Returns a video image and mask plot
    :param image: video image
    :param mask:  binary mask obtained by wave breaking method
    :param beach_path: path of beach folder
    :param name: image name
    :param size: dictionary with information about new and old sizes
    :return: plot
    """

    fig, ax = plt.subplots(1, 3, figsize=(24, 10))
    image = cv2.resize(image, (size['new_width'], size['new_height']), interpolation=cv2.INTER_AREA)
    # Figure 1
    ax[0].imshow(image.squeeze(), cmap='gray')
    ax[0].grid(False)
    ax[0].set_title('Video image')
    ax[0].set_xlabel('Cross-shore Distance [pixel]')
    ax[0].set_xlabel('Alongshore Distance [pixel]')

    # Figure 2
    ax[1].imshow(mask.squeeze(), cmap='gray')
    ax[1].grid(False)
    ax[1].set_title('Prediction mask')
    ax[1].set_xlabel('Cross-shore Distance [pixel]')

    # Figure 4
    ax[2].imshow(image.squeeze(), cmap='gray')
    ax[2].contour(mask.squeeze(), colors='r', levels=[0.1])
    ax[2].grid(False)
    ax[2].set_title('Prediction over image')
    ax[2].set_xlabel('Cross-shore Distance [pixel]')

    plt.savefig('.'+beach_path + '/plot_results/' + name,
                bbox_inches='tight',
                pad_inches=1)
