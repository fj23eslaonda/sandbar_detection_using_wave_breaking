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


# -----------------------------------------------------------------
# 
# INPUTS
#
# -----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Path and image size')
    parser.add_argument('--beach_path', type=str,
                        help='beach folder')
    parser.add_argument('--main_path', type=str, default=os.getcwd(),
                        help='Duck model main path')
    parser.add_argument('--image_path', type=str, default='./frames/',
                        help='Image input path')
    parser.add_argument('--output_path', type=str, default='./prediction_mask',
                        help='Output path for mask')
    parser.add_argument('--plot_mask', type=bool, default=False,
                        help='Output path for mask')
    parser.add_argument('--plot_mask_over_img', type=bool, default=False,
                        help='Output path for video and mask')
    parser.add_argument('--orientation', type=str, default='vertical',
                        help='Wave directions')
    parser.add_argument('--number_img', type=int, default = False,
                        help='Wave directions')
    arguments = parser.parse_args()
    return arguments


# -----------------------------------------------------------------
# 
# RUN MODEL
#
# -----------------------------------------------------------------
# INPUTS
args = parse_args()
main_path          = args.main_path
beach_path         = args.beach_path
image_path         = args.image_path
output_path        = args.output_path
plot_mask_over_img = args.plot_mask_over_img
plot_mask          = args.plot_mask
orientation        = args.orientation
number_img         = int(args.number_img)

# -----------------------------------------------------------------
#
# REMOVE EXISTENT FOLDERS
#
# -----------------------------------------------------------------
if os.path.exists(args.main_path + '/name_img.txt'):
    os.remove(args.main_path + '/name_img.txt')
# -----------------------------------------------------------------
if os.path.exists('.' + args.beach_path + args.output_path):
    shutil.rmtree('.' + args.beach_path + args.output_path)
# -----------------------------------------------------------------
if os.path.exists('.' + args.beach_path + '/plot_results/'):
    shutil.rmtree('.' + args.beach_path + '/plot_results/')
# -----------------------------------------------------------------
if os.path.exists('.' + args.beach_path + '/sandbar_results/'):
    shutil.rmtree('.' + args.beach_path + '/sandbar_results/')
# -----------------------------------------------------------------
if plot_mask_over_img:
    os.system('mkdir -m 771 .' + args.beach_path + '/plot_results/')
if plot_mask:
    os.system('mkdir -m 771 .' + args.beach_path + args.output_path)
os.system('mkdir -m 771 .' + args.beach_path + '/sandbar_results/')
# -----------------------------------------------------------------

# FUNCTIONS
model = duck_model.DuckModel(main_path,
                             beach_path,
                             image_path,
                             output_path,
                             plot_mask_over_img,
                             plot_mask,
                             orientation,
                             number_img
                             )
# ITERATION
model.run_model()
