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
import json


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
def plot_predictions(image, mask, mask_over_image_folder, name, size):
    """
    Returns a video image and mask plot
    :param image: video image
    :param mask:  binary mask obtained by wave breaking method
    :param mask_over_image_folder: folder to save plots
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

    plt.savefig(f"{mask_over_image_folder / name}",
                bbox_inches='tight',
                pad_inches=1)


def read_json_to_dict(filename):
    """

    Parameters
    ----------
    filename : string
        name of the file to load

    Returns
    -------
    dict : dictionary containing numpy arrays
    """
    try:
        with open(filename, 'r') as infile:
            dictionary = json.load(infile)
            for k in dictionary.keys():
                if type(dictionary[k]) is list:
                    dictionary[k] = np.array(dictionary[k])
            return dictionary
    except IOError:
        raise
