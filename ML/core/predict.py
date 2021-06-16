import os
import numpy as np
import tensorflow as tf
import multiprocessing
from dynaconf import settings
settings.load_file(path='ML/core/config.py')

from ML.core.utils import (get_wav, to_mfcc, load_categories,
                    normalize_mfcc, make_segments, 
                    segment_one, load_data, get_input_shape)
                            
from ML.core.model import Model

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Accent_Identification:
    def __init__(self):
        # # load categories
        self.categories = load_categories('ML/core/categories/labels.json')
        
        # # initiate model
        self.model = Model(input_shape = (128, 30, 1), num_classes = len(self.categories), lr = settings.LR)

        #load the weights                
        try:
            self.model.load_weights(settings.LOAD_CHECKPOINT_DIR)
        except:
            logging.error('Unable to load checkpoints')
            

    def predict(self, path):
        #load DATA
        X = get_wav(path)

        X = to_mfcc(X)

        X = normalize_mfcc(X)

        X = segment_one(X, COL_SIZE = settings.COL_SIZE)

        prediction = self.model.predict(X)
        prediction = np.argmax(prediction, axis = 1)
        print(prediction)
        prediction = np.bincount(prediction)
        prediction = np.argmax(prediction)
        prediction = [key for key, value in self.categories.items() if value == prediction]
        print(prediction)

        return prediction
            




