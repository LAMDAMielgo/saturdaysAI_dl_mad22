# MODULE WHERE ALL THE FUNCTIONS
# THAT GET DATA FROM SOURCE AND 
# TRANSFORMS THEM INTO PREPROCESSING VARIABLES
#

import glob
import numpy as np

from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import StandardScaler

from PIL import Image

# ---------------------------------------------------------------------------------
# globals

def img_df(
        image_path:str, 
        shape:tuple
    ) -> list:
    """ Parsed the image given through its path and retuns its tensor
    representation 

    Args:
        image_path (str): relative path where the image is
        shape (tuple): default value is the global for the MobileNet

    Returns:
        list: array of the image
    """
    image = Image.open(image_path)
    image_resized = image.resize(shape, Image.ANTIALIAS)
    img_array = np.asarray(image_resized)

    return img_array


def transform_X(datadict, img_size):
    """ 
    Transformation of imgs into a normalized tensor
    Normalization criteria based on the RBG vector ranges

    Args:
        datadict (dict): input dict with ['id']['values'] as values
             and imagepath as keys
        img_size (tuple): global with input image size to consider
    Return:

    """
    norm_x = lambda img: img / 255.

    X = {i: img_df(k, shape=img_size) for i, k in enumerate(datadict.keys())}
    X = np.array(list(map(norm_x, X.values())))

    X = preprocess_input(X)

    print(f"X as {type(X)} with shape {X.shape}")
    return X


def transform_Y(datadict):
    """ 
    Transformation of target regression values into a normalized vector
    Normalization criteria based on percentages.
    """ 
    scaler = StandardScaler()
    norm_y = lambda d: list(map(lambda v: int(v) / 100., d.values()))

    Y = {i: datadict[k]['values'] for i, k in enumerate(datadict.keys())}
    Y = np.array(list(map(norm_y, Y.values())))

    Y = scaler.fit_transform(Y)

    print(f"Y as {type(Y)} with shape {Y.shape}")
    return Y


# ---------------------------------------------------------------------------------
# end