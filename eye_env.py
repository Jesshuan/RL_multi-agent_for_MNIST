
import numpy as np


class Eye():

    def __init__(self):

        self.dim_output = 8 # fixed by aggregation processus

    def aggregate_box(self, box_sample):

        box_length = box_sample.shape[1]
        box_height = box_sample.shape[0]

        # Contrast box generation
        contrast_box = np.zeros((box_height, box_length), dtype=np.float16)
        for i in range(box_height):
            for j in range(box_length):
                pixel_value = box_sample[i, j]
                start_box_vertical = max(0, i - 1)
                end_box_vertical = min(i + 1, box_height - 1)
                start_box_horizontal = max(0, j - 1)
                end_box_horizontal = min(j + 1, box_length - 1)
                conv_box = box_sample[start_box_vertical:end_box_vertical + 1, start_box_horizontal:end_box_horizontal + 1]
                contrast_box[i, j] = np.sqrt(np.sum((conv_box - pixel_value)**2))
                """conv_size = (end_box_horizontal - start_box_horizontal +1 )*(end_box_vertical - start_box_vertical +1 )
                tot = np.sum(conv_box)
                contrast_box[i, j] = pixel_value*(conv_size - tot) + (1- pixel_value)*tot"""

        # Aggregation
        box_min = np.min(contrast_box)
        box_max = np.max(contrast_box)
        box_mean = np.mean(contrast_box)
        box_std = np.round(np.std(contrast_box), 2)
        diff_hd_mean = np.mean(contrast_box[:box_height//2, :]) - np.mean(contrast_box[box_height//2 :, :])
        diff_lr_mean = np.mean(contrast_box[:, :box_length//2] - np.mean(contrast_box[:, box_length//2:]))
        diff_hd_max = np.max(contrast_box[:box_height//2, :]) - np.max(contrast_box[box_height//2 :, :])
        diff_lr_max = np.max(contrast_box[:, :box_length//2] - np.max(contrast_box[:, box_length//2:]))

        return np.array([box_min, box_max, box_mean, box_std, diff_hd_mean, diff_lr_mean, diff_hd_max, diff_lr_max], dtype=np.float16)




class BlurEye(Eye):

    def __init__(self, blur_eye_dim, associated_clear_eye_dim, array_image, blur_factor_reduction=2):

        super(BlurEye, self).__init__()

        self.blur_eye_dim = blur_eye_dim
        self.associated_clear_eye_dim = associated_clear_eye_dim
        self.blur_factor_reduction = blur_factor_reduction
        self.array_image = self.add_a_blur_padding(array_image)
        self.length = array_image.shape[1]
        self.height = array_image.shape[0]

        self.blur_matrix = self.compute_conv_matrix()

    def add_a_blur_padding(self, arr_img):
        pad_w = self.blur_eye_dim // 2 - self.associated_clear_eye_dim // 2
        return np.pad(arr_img, pad_width=pad_w, mode='constant', constant_values=0)


    def reduce_by_mean(self, img_box, nb_split=2):
        return np.array([np.apply_over_axes(np.mean, sub_img, axes=(0, 1)) \
                         for img in np.array_split(img_box, nb_split) \
                            for sub_img in np.array_split(img, nb_split, 1)])\
                                .reshape(nb_split, -1, nb_split).squeeze()

    def compute_conv_matrix(self):

        nb_split = self.blur_eye_dim // self.blur_factor_reduction

        blur_matrix = np.zeros((self.height - self.associated_clear_eye_dim + 1, self.length - self.associated_clear_eye_dim + 1, self.dim_output), dtype=np.float16)

        for i in range(self.height - self.associated_clear_eye_dim + 1):
            for j in range(self.length - self.associated_clear_eye_dim + 1):
                box_sample = self.array_image[i : i + self.blur_eye_dim, j: j + self.blur_eye_dim]
                box_sample = self.reduce_by_mean(box_sample, nb_split=nb_split)
                blur_matrix[i][j] = self.aggregate_box(box_sample)

        return blur_matrix
    

class ClearEye(Eye):

    def __init__(self, clear_eye_dim: int, array_image):

        super(ClearEye, self).__init__()

        self.clear_eye_dim = clear_eye_dim

        self.array_image = array_image
        self.length = array_image.shape[1]
        self.height = array_image.shape[0]

        self.clear_matrix = self.compute_conv_matrix()

    def compute_conv_matrix(self):

        clear_matrix = np.zeros((self.height - self.clear_eye_dim + 1, self.length - self.clear_eye_dim + 1, self.dim_output), dtype=np.float16)

        for i in range(self.height - self.clear_eye_dim + 1):
            for j in range(self.length - self.clear_eye_dim + 1):
                box_sample = self.array_image[i : i + self.clear_eye_dim, j: j + self.clear_eye_dim]
                clear_matrix[i][j] = self.aggregate_box(box_sample)

        return clear_matrix


class OneImageEnv():

    def __init__(self,
                 array_image,
                 clear_eye_dim=4,
                 blur_eye_dim=16,
                 blur_factor_reduction=2):
        
        if clear_eye_dim %2 != 0:
            raise ValueError("Sorry, but 'clear eye dim' must be a multiple of 2.")
        
        if blur_eye_dim %2 != 0:
            raise ValueError("Sorry, but 'clear eye dim' must be a multiple of 2.")
        
        self.clear_eye_dim = clear_eye_dim
        self.blur_eye_dim = blur_eye_dim
        self.blur_facotr_reduction = blur_factor_reduction

        self.array_image = array_image
        self.length = array_image.shape[0]
        self.height = array_image.shape[1]

        self.eye_position = [np.random.randint(0, self.length - clear_eye_dim), np.random.randint(0, self.height - clear_eye_dim)]

        self.clear_eye = ClearEye(clear_eye_dim=clear_eye_dim,
                                  array_image=array_image)
        self.blur_eye = BlurEye(blur_eye_dim=blur_eye_dim,
                                associated_clear_eye_dim=clear_eye_dim,
                                array_image=array_image,
                                blur_factor_reduction=blur_factor_reduction)