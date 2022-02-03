"""
PURPOSE   : Convert images to matrix, make a prediction and convert matrix to images.
AUTHOR    : Francisco Sáez R.
EMAIL     : francisco.saez@sansano.usm.cl
V1.0      : 22/10/2021
V2.0      : 26/11/2021

This code allow transforming images from png or jpg file to matrix to
make a prediction with Duck Model proposed for Sáez et al. (2021)
"""

# -----------------------------------------------------------------
#
# PACKAGES
#
# -----------------------------------------------------------------
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from all_functions import *


class DuckModel:

    # -----------------------------------------------------------------
    # 
    # PARAMETERS
    #
    # -----------------------------------------------------------------
    def __init__(self,
                 all_inputs,
                 number_img):

        self.beach_folder = all_inputs['main_path'] / all_inputs['beach_folder']
        self.images_folder = self.beach_folder / all_inputs['image_folder']
        self.model_folder = all_inputs['main_path'] / "model"
        self.save_path = self.beach_folder / all_inputs['sandbar_results']
        self.number_img = number_img
        self.orientation = all_inputs['orientation']
        self.plot_mask = all_inputs['plot_mask']
        self.plot_mask_over_image = all_inputs['plot_mask_over_image']
        self.mask_folder = self.beach_folder / all_inputs['mask_folder']
        self.mask_over_image_folder = self.beach_folder / all_inputs['mask_over_image_folder']
        self.beach_name = all_inputs['beach_name']

    # -----------------------------------------------------------------
    # 
    # LOAD IMAGE NAMES LIST
    #
    # -----------------------------------------------------------------

    def load_list_img(self):
        """
        return a list of all name images contain in images folder
        :param: beach folder path and image folder path
        :return: list
        """
        list_img = sorted(os.listdir(self.images_folder))

        return list_img

    # -----------------------------------------------------------------
    # 
    # LOAD DUCK MODEL
    #
    # -----------------------------------------------------------------
    def load_model(self):
        """
        Return U-Net model propose for Sáez et al. (2021)
        -----
        model is a groups layers into an object.
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        -----
        :return: model
        """
        # Load JSON and Create model

        # Load JSON and Create model

        json_file = open(self.model_folder / "model_final.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, {"tf": tf})
        # Load weight
        model.load_weights(self.model_folder / "best_model_final.h5")
        return model

    # -----------------------------------------------------------------
    # 
    # GET SIZE
    #
    # -----------------------------------------------------------------
    def new_size(self, list_img):
        """
        Return four int values:
        - new height and new width used to rescale original image into a new image
          used to make prediction.
        - old height and old width used to rescale predicted image into original size.

        Example
        -------
        old_height = 1000
        old_width  = 2000

        old_height % 512 = 464 >= 256
            ----> new_height = 512*(old_height // 512 + 1)
            ----> new_height = 512*(1 + 1)
            ----> new_height = 1024

        old_width % 512 = 488  >= 256
            ----> new_width = 512*(old_width // 512 + 1)
            ----> new_width = 512*(3 + 1)
            ----> new_width = 2048

        new_height = 1024, new_width = 2048, old_height = 1000, old_width = 2000
        :param list_img: list of image name
        :return: new_height, new_width, old_height, old_width
        """
        size = dict()
        # INPUTS
        img = cv2.imread(f"{self.images_folder / list_img[0]}", 0)
        old_height, old_width = (img.squeeze()).shape
        # ------------------------------------------------------------------------------------
        # CALCULATE NEW HEIGHT
        if old_height % 512 >= 256:
            new_height = 512 * (old_height // 512 + 1)
        else:
            new_height = 512 * (old_height // 512)
        # ------------------------------------------------------------------------------------
        # CALCULATE NEW WIDTH
        if old_width % 512 >= 256:
            new_width = 512 * (old_width // 512 + 1)
        else:
            new_width = 512 * (old_width // 512)
        # ------------------------------------------------------------------------------------
        size['new_width'] = new_width
        size['new_height'] = new_height
        size['old_width'] = old_width
        size['old_height'] = old_height

        return size

    # -----------------------------------------------------------------
    # 
    # SPLIT IMAGES
    #
    # -----------------------------------------------------------------
    def split_img(self, name_img, size, mean_image):
        """
        Return a dictionary with square images of 512x512 in size to make the prediction.

        :param name_img: name image to split
        :param size: dictionary with new and old size
        :param mean_image: matrix to save each image and create average image.
        :return: x_tst and mean_image
        """
        # READ IMG AND SAVE IT
        original_img = cv2.imread(f"{self.images_folder / name_img}")
        mean_image = mean_image + original_img

        # PROCESSING
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size['new_width'], size['new_height']), interpolation=cv2.INTER_AREA)
        # -------------------------------------------------------------------------------------
        # Y DIRECTION SPLIT
        n_split_y = size['new_height'] // 512
        split_y_dir = np.concatenate(np.vsplit(img, n_split_y), axis=1)

        # X DIRECTION SPLIT
        n_split_x = np.shape(split_y_dir)[1] // 512
        split_x_dir = np.hsplit(split_y_dir, n_split_x)

        x_tst = split_x_dir
        return x_tst, mean_image, original_img

    # -----------------------------------------------------------------
    #
    # PREDICTION
    #
    # -----------------------------------------------------------------
    def get_prediction(self, model, x_tst):
        """
        Returns a dictionary with all masks obtained from the split images.

        Example
        -------
        x_tst = dict{ 'part_1' : array of 512x512x1 in size,
                      'part_2' : array of 512x512x1 in size,
                      ...
                      'part_N' : array of 512x512x1 in size,
        }

        mask = dict()

        for part_i in x_tst.keys():
            mask[mask_i] = model.predict(x_tst[part_i])

        :param model: U-Net model loaded using load_model function
        :param x_tst: all split images used to make prediction
        :return: mask ---> dictionary
        """
        # INPUTS
        masks = list()

        # MAKE PREDICTION IN EACH SPLIT IMAGES
        for img in x_tst:
            X = np.zeros((1, 512, 512, 1))
            if self.orientation == 'vertical':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            X[0] = img[..., np.newaxis] / 255.0

            # CONDITION TO JUST USE IMAGE WITH INFORMATION

            if np.mean(X) >= 10 / 255:

                y_tst = np.squeeze(model.predict(X, verbose=True)) > 0.7

                if self.orientation == 'vertical':
                    y_tst = cv2.rotate(y_tst * 255.0, cv2.ROTATE_90_COUNTERCLOCKWISE) / 255.0
                masks.append(y_tst)
            else:
                masks.append(np.zeros(np.squeeze(X[0]).shape))
        return masks

    # -----------------------------------------------------------------
    # 
    # CONCATENATE
    #
    # -----------------------------------------------------------------
    def concatenate_and_save(self, masks, name_img, mean_mask, size):
        """
        Returns a cumulative mask matrix. In addition, this function allows to concatenate all split masks.

        :param masks: list with all mask
        :param name_img: name image
        :param mean_mask: matrix to save each mask and generate cumulative breaking map
        :param size: dictionary with new and old size
        :return: mean_mask
        """
        # CONCATENATE ALL SQUARE IMAGE IN A BIG ROW
        masks = np.concatenate(tuple(masks), axis=1)

        # SPLIT BIG ROW
        split_x_dir = np.hsplit(masks, masks.shape[1] // size['new_width'])

        # GET IMAGE AND RESCALE WITH ORIGINAL SIZE
        new_mask = np.concatenate(tuple(split_x_dir), axis=0)

        # IF YOU WANT SAVE MASK AS IMAGE
        if self.plot_mask:
            cv2.imwrite(f"{self.mask_folder / name_img}", new_mask * 255)

        mean_mask = mean_mask + new_mask
        return mean_mask, new_mask

    # -----------------------------------------------------------------
    # 
    # PLOT MESSAGES
    #
    # -----------------------------------------------------------------
    @staticmethod
    def plot_messages(message):
        """
        Returns a message to find out which part of code is running
        :param message: str
        :return: print message
        """
        print('#---------------------------------------------')
        print(' ')
        print(message)
        print(' ')

    # -----------------------------------------------------------------
    #
    # PLOT FINAL RESULT
    #
    # -----------------------------------------------------------------
    def plot_img_and_mask(self, mean_img, mean_mask, x_point, y_point):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        plt.suptitle('The Sand Bar detection result on ' + self.beach_name, fontsize=20)

        axs[0].imshow(mean_img / np.max(mean_img))
        map1 = axs[0].imshow(mean_mask, vmax=np.nanmax(mean_mask), cmap='turbo')
        if self.orientation == 'vertical':
            axs[0].set_xlabel('Alongshore distance, y [pixels]', fontsize=14)
            axs[0].set_ylabel('Cross-shore distance, x [pixels]', fontsize=14)
        else:
            axs[0].set_ylabel('Alongshore distance, y [pixels]', fontsize=14)
            axs[0].set_xlabel('Cross-shore distance, x [pixels]', fontsize=14)
        axs[0].set_title('Cumulative Breaking Map', fontsize=14)
        axs[0].grid(linestyle='--', alpha=0.7)
        fig.colorbar(map1, ax=axs[0], fraction=0.05, pad=0.04)

        axs[1].imshow(mean_img / np.max(mean_img), vmax=np.max(mean_img))
        if self.orientation == 'vertical':
            axs[1].scatter(x_point, y_point, marker='x', c='r', s=8, label='Wave Breaking Method')
            axs[1].set_xlabel('Alongshore distance, y [pixels]', fontsize=14)
        else:
            axs[1].scatter(y_point, x_point, marker='x', c='r', s=8, label='Wave Breaking Method')
            axs[1].set_xlabel('Cross-shore distance, x [pixels]', fontsize=14)

        axs[1].set_title('Average Image', fontsize=14)
        axs[1].grid(linestyle='--', alpha=0.7)
        axs[1].legend()

        plt.savefig(f"{self.save_path}" + '/cumulative_breaking.png',
                    bbox_inches='tight',
                    pad_inches=1)
        plt.show()
        plt.close("all")

        # SAVE MATRIX
        np.save(f"{self.save_path}" + '/mean_image_' + self.beach_name + '.npy', mean_img)
        np.save(f"{self.save_path}" + '/mean_mask_' + self.beach_name + '.npy', mean_mask)

    # -----------------------------------------------------------------
    # 
    # RUN MODEL
    #
    # -----------------------------------------------------------------
    def run_model(self):
        # IMAGES NAME
        if self.number_img:
            list_img = self.load_list_img()[:self.number_img]
        else:
            list_img = self.load_list_img()

        # LOAD MODEL
        self.plot_messages('LOAD MODEL')
        model = self.load_model()

        # CREATE MATRIX TO PLOT
        size = self.new_size(list_img)
        # MEAN IMAGE MATRIX AND MASK MATRIX
        mean_image = np.zeros((size['old_height'], size['old_width'], 3))
        mean_mask = np.zeros((size['new_height'], size['new_width']))

        for ix, name_img in enumerate(list_img):
            self.plot_messages('PREDICTION N°' + str(ix + 1) + ' of ' + str(len(list_img)))
            # ---------------------------------------------------------
            x_tst, mean_image, original_img = self.split_img(name_img, size, mean_image)

            # ---------------------------------------------------------
            masks = self.get_prediction(model, x_tst)
            # ---------------------------------------------------------
            mean_mask, mask = self.concatenate_and_save(masks, name_img, mean_mask, size)

            if self.plot_mask_over_image:
                plot_predictions(original_img, mask, self.mask_over_image_folder, name_img, size)
        # ---------------------------------------------------------
        # INTERPOLATION
        mean_mask = interp_cumulative_breaking(mean_mask, size)
        x_point, y_point, mean_mask = identify_sandbar_pts(mean_mask, self.orientation)
        # ---------------------------------------------------------
        # SAND BAR
        self.plot_img_and_mask(original_img, mean_mask, x_point, y_point)
