
import numpy as np

from mnist_eye_environment.eyes import BlurEye, ClearEye


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