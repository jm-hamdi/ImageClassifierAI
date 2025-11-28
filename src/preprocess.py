import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH

def preprocess_data(x_train, x_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # reshape for CNN
    x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    return x_train, x_test
