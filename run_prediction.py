"""
PROGRAM   : run_prediction.py
PURPOSE  : Convert images to matrix, make a prediction and convert matrix to images.
AUTHOR    : Francisco Sáez R.
EMAIL     : francisco.saez@sansano.usm.cl
V1.0      : 22/10/2021

This code allow transforming images from png or jpg file to matrix to
make a prediction with Duck Model proposed for Sáez et al. (2021)
"""

# -----------------------------------------------------------------
#
# PACKAGES
#
# -----------------------------------------------------------------
import os
import argparse
import duck_model
import shutil
from pathlib import Path

from all_functions import *

# -----------------------------------------------------------------
# 
# INPUTS
#
# -----------------------------------------------------------------
# Read file
parser = argparse.ArgumentParser(description='All necessary inputs and number of images to use')
parser.add_argument('--parameters', type=str,
                    help='Dictionary with all necessary inputs')
parser.add_argument('--number_img', type=int, default=False,
                    help='Number of images to use')
args = parser.parse_args()

# -----------------------------------------------------------------
# 
# RUN MODEL
#
# -----------------------------------------------------------------
# INPUTS
# open file
all_inputs = read_json_to_dict(args.parameters)
all_inputs['main_path'] = Path(os.getcwd())

number_img = args.number_img
# ------------------------------------------------------------
#
# REMOVE EXISTENT FOLDERS
#
# -----------------------------------------------------------------
if (all_inputs['main_path'] / all_inputs['mask_folder']).exists():
    shutil.rmtree(all_inputs['main_path'] / all_inputs['mask_folder'])
# -----------------------------------------------------------------
if (all_inputs['main_path'] / all_inputs['mask_over_image_folder']).exists():
    shutil.rmtree(all_inputs['main_path'] / all_inputs['mask_over_image_folder'])
# -----------------------------------------------------------------
if (all_inputs['main_path'] / all_inputs['sandbar_results']).exists():
    shutil.rmtree(all_inputs['main_path'] / all_inputs['sandbar_results'])
# -----------------------------------------------------------------
if all_inputs['plot_mask']:
    (all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['mask_folder']).mkdir(mode=0o755,
                                                                                             exist_ok=True)

if all_inputs['plot_mask_over_image']:
    (all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['mask_over_image_folder']).mkdir(mode=0o755,
                                                                                                        exist_ok=True)

(all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['sandbar_results']).mkdir(mode=0o755,
                                                                                             exist_ok=True)
# -----------------------------------------------------------------

# FUNCTIONS
model = duck_model.DuckModel(all_inputs,
                             number_img)
# ITERATION
model.run_model()
