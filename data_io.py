import os
import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
import sklearn

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
VALIDATION_RATIO = 0.3

def get_data(batch_size, epochs):
    # Load from file
    data = load_dataset()

    # Split data
    train_samples, validation_samples = train_test_split(data, test_size=VALIDATION_RATIO)

    return train_samples, validation_samples, (len(train_samples)*6), len(validation_samples)*6

def load_dataset():
    log_file_original = os.path.join(DATA_PATH, 'driving_log.csv')
    log_file_centerDriving = os.path.join(DATA_PATH,'centerDrivingLong','driving_log.csv')
    log_file_recovery = os.path.join(DATA_PATH, 'recoveryLong', 'driving_log.csv')
    log_file_split = os.path.join(DATA_PATH, 'driving_log_split.csv')

    if os.path.exists(log_file_centerDriving) and os.path.exists(log_file_recovery):
        original = pd.read_csv(log_file_original)
        centerDriving = pd.read_csv(log_file_centerDriving)
        recovery = pd.read_csv(log_file_recovery)
        data = original.append(centerDriving.append(recovery, ignore_index=True), ignore_index=True)
        n = len(data)
        print('\nDataset has {} samples'.format(n))
        data.to_csv(log_file_split, index=False)

    return data


def load_image(file_path):
    img = imread(os.path.join(DATA_PATH, file_path.strip()))
    return img

def append_image(x, y, img, steering, flip=0):
    if flip == 1:
        img = np.fliplr(img)
        steering = -steering

    x.append(img)
    y.append(steering)

def append_side_image(x, y, steering, img_left, img_right):
    # Augment with left and right image
    #   add flip
    _correction = 0.5

    append_image(x, y, img_left, steering + _correction, 0)
    append_image(x, y, img_left, steering - _correction, 1)
    append_image(x, y, img_right, steering - _correction, 0)
    append_image(x, y, img_right, steering + _correction, 1)

def generator(samples, batch_size=64, input_shape=(160, 320, 3)):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Shuffle data
        samples.reindex(np.random.permutation(samples.index))
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for idx in batch_samples.index:
                center_image = load_image(batch_samples.loc[idx, 'center'])
                center_angle = batch_samples.loc[idx, 'steering']
                append_image(images, angles, center_image, center_angle, 0)
                append_image(images, angles, center_image, center_angle, 1)

                img_left = load_image(batch_samples.loc[idx, 'left'])
                img_right = load_image(batch_samples.loc[idx, 'right'])
                append_side_image(images, angles, center_angle, img_left, img_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)