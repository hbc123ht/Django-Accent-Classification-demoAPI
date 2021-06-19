from django.apps import AppConfig
from django.conf import settings
import tensorflow as tf
import numpy as np

class Predictor(AppConfig):
    """
    Predictor app
    """
    name = 'ML'
    model = tf.keras.models.load_model(settings.MODEL_PATH)


