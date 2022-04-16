# MODULE WHERE ALL THE FUNCTIONS
# THAT GET DATA FROM SOURCE AND 
# TRANSFORMS THEM INTO PREPROCESSING VARIABLES
#

import glob
import numpy as np
import os
import re
import requests

from pprint import pprint

# ---------------------------------------------------------------------------------
# globals

# The exercises downloads imgs from this paper and stores them in TARGET_FOLDER
URL = 'http://crowley-coutaz.fr/HeadPoseDataSet/HeadPoseImageDatabase.tar.gz'
TARGET_FOLDER = "/content/drive/MyDrive/satAI/week_03/challenge"

DATA_FOLDER = "./data"

# Patterns used to parse strings
GLOB_PATTERN = f"{DATA_FOLDER}/Person*/*.txt"
REGEX_PATTERN = "person(?P<code>[0-9]{5})(?P<t>[+|-][0-9]{,2})(?P<p>[+|-][0-9]{,2})"

# ---------------------------------------------------------------------------------
# functions

def get_data(
        url:str = URL,
        target_folder:str = TARGET_FOLDER
    ) -> str:
    """ 
    Get data from url and saves it into target_folder

    Args:
        url (str): webpath where a compress file is stored
        target_folder (str): target Drive location to save the file

    Return:
        str: file path where it has been stored.
    """

    res = requests.get(url)
    name = re.findall("(?<=t/)(.*)(?=.tar.gz)", url)[0]  

    if res.status_code == 200:

        fpath = f"{target_folder}{name}"        
        with open(fpath, 'wb') as f: f.write(res.content)        
        
        return fpath

    else:
        print(res.status_code)


def parse_txtfile(txtf:str) -> tuple:
    """ Open the text file and reads its content which comes in the following structure:
        ----------------------------------
            0 [Corresponding Image File]   tuple(0)
            1         
            2 Face
            3 [Face Center X]              tuple(1)
            4 [Face Center Y]
            5 [Face Width]
            6 [Face Height]
        ----------------------------------
    Args:
        txtf (str): text file path to read

    Return:
        tuple: tuple with image file path name and values to add
    """
    str_strip = lambda l: list(map(lambda s: s.strip(), l))

    with open(txtf) as f:
        txt_info = str_strip(f.readlines())

    return txt_info[0], txt_info[3:]


def construct_targetvalues():
    """ This function iters through all the text files and constructs
    a dictionary such as:

        [image_path_name] : {
            'id'     : tuple with ( [Id], [Serie], [Number] )
            'values' : dictionary with target vector as dict 
        }

    Args:
        None

    Return:
        dict: dictionary with necessary information ordered
    """
    # Variables
    extract_classes = lambda s: (s[:2], s[2], s[3:])
    imgs_dict = dict()

    def init_dict():
        dglobals = {'id': tuple,'values': dict()}

        d = dict().fromkeys(dglobals.keys())
        for k, dtype in dglobals.items(): d[k] = dtype
        
        return d


    for txt_path in glob.glob(GLOB_PATTERN):

        # read text and compose img path
        img_name, txt_values = parse_txtfile(txt_path)    
        img_path = os.path.join( 
            txt_path.rsplit(os.sep, maxsplit=1)[0],
            img_name
        )

        # initalize dict at imagepath and assing data types
        imgs_dict[img_path] = init_dict()

        # use of regex
        regx  = re.compile(REGEX_PATTERN)
        match = re.match(regx, img_name)

        # update values with angles and text file information already read
        imgs_dict[img_path]['values'].update(
            **{ 'T': match['t'], 'P': match['p'] },
            **{ k:v for k,v in zip(['X', 'Y', 'W', 'H'], txt_values)}
        )
        imgs_dict[img_path]['id']= extract_classes(match['code'])
        # end loop

    pprint(imgs_dict[img_path])
    return imgs_dict


# ---------------------------------------------------------------------------------
# end