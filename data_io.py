import os
import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
import sklearn

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
VALIDATION_RATIO = 0.3

def get_data(batch_size):
    # Load from file
    data = load_dataset()

    # Filter large delta in steering
    data = filter_steering(data)

    # Split data
    train_samples, validation_samples = train_test_split(data, test_size=VALIDATION_RATIO)

    ratio = 0.1
    return train_samples, validation_samples, len(train_samples)*ratio, len(validation_samples)*ratio

def load_dataset():
    log_file_original = os.path.join(DATA_PATH, 'driving_log.csv')
    log_file_centerDriving = os.path.join(DATA_PATH,'centerDriving','driving_log.csv')
    log_file_recovery = os.path.join(DATA_PATH, 'recoveryLong', 'driving_log.csv')
    log_file_split = os.path.join(DATA_PATH, 'driving_log_split.csv')

    if os.path.exists(log_file_centerDriving) and os.path.exists(log_file_recovery):
        original = pd.read_csv(log_file_original)
        centerDriving = pd.read_csv(log_file_centerDriving)
        recovery = pd.read_csv(log_file_recovery)
        data = original.append(recovery, ignore_index=True)
        n = len(data)
        print('\nDataset has {} samples'.format(n))
        data.to_csv(log_file_split, index=False)

    return data


def load_image(file_path):
    img = imread(os.path.join(DATA_PATH, file_path.strip()))
    # Crop to 80, 160 - replaced with Keras.Cropping2D
    #img = imresize(img, (80, 160, 3))
    #Normalize
    return (img / 255.0-0.5)

def filter_steering(data, percentage=8, threshold = 0.5):
    #
    # Filtering % steering angles with delta above threshold
    #    default is 70% of all angles greater than 0.5
    #
    rows = []
    delta = 0
    previousSteering = data.loc[0, 'steering']
    ind = data.index.tolist()
    for i in ind:
        if i > 0:
            delta = np.absolute(previousSteering - data.loc[i, 'steering'])
        random = np.random.randint(10)
        if (delta > threshold) and (random < percentage):
            rows.append(i)

    data = data.drop(data.index[rows])
    print("Filtered {} rows with large delta steering".format(len(rows)))

    return data

def add_image(x, y, img, steering, i, flip=0):
    if flip == 1:
        x[i, :, :, :] = np.fliplr(img)
        y[i, 0] = -steering
    else:
        x[i, :, :, :] = img
        y[i, 0] = steering
    return i + 1

def add_side_image(x, y, steering, i, batch_size, img_left, img_right):
    # Augment with slight increase and decrease for left and right image
    #   hard increase opposite direction camera
    #   soft decrease current direction camera
    #   add flip
    _hard = 0.15
    _soft = -0.025

    # left turn
    if steering < 0:
        if i < batch_size:
            i = add_image(x, y, img_left, steering + _soft, i)
        if i < batch_size:
            i = add_image(x, y, img_left, steering + _soft, i, 1)
        if i < batch_size:
            i = add_image(x, y, img_right, steering + _hard, i)
        if i < batch_size:
            i = add_image(x, y, img_right, steering + _hard, i, 1)
    # right turn
    else:
        if i < batch_size:
            i = add_image(x, y, img_right, steering + _soft, i)
        if i < batch_size:
            i = add_image(x, y, img_right, steering + _soft, i, 1)
        if i < batch_size:
            i = add_image(x, y, img_left, steering + _hard, i)
        if i < batch_size:
            i = add_image(x, y, img_left, steering + _hard, i, 1)
    return i

def generate_valid(data, batch_size=64, input_shape=(160, 320, 3)):

    # Shuffle data
    data = data.reindex(np.random.permutation(data.index))
    num_samples = len(data)

    for offset in range(0, num_samples, batch_size):
        batch_samples = data[offset:offset + batch_size]

        while 1:
            x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
            y = np.zeros((batch_size, 1))
            i = 0

            while i < batch_size:
                idx = np.random.choice(batch_samples.index, 1, replace=False)[0]
                img = load_image(batch_samples.loc[idx, 'center'])
                steering = batch_samples.loc[idx, 'steering']

                i = add_image(x, y, img, steering, i, 0)
            yield sklearn.utils.shuffle(x, y)

def generate_train(data, batch_size=64, input_shape=(160, 320, 3)):
    #
    # Generate data with augmentation, filtering
    #

    # Shuffle data
    data = data.reindex(np.random.permutation(data.index))
    num_samples = len(data)

    for offset in range(0, num_samples, batch_size):
        batch_samples = data[offset:offset + batch_size]

        while 1:
            x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
            y = np.zeros((batch_size, 1))
            i = 0

            while i < batch_size:
                idx = np.random.choice(batch_samples.index, 1, replace=False)[0]
                img = load_image(batch_samples.loc[idx, 'center'])
                steering = batch_samples.loc[idx, 'steering']

                i = add_image(x, y, img, steering, i, 0)

                if i < batch_size:
                    # Horizontally flip the image
                    i = add_image(x,y, img, steering, i, 1)

                # Augment with left and right image
                #    with slight steering angle adjustment
                #    with horizontally flipped image
                random = np.random.randint(10)
                if random < 8 and np.absolute(steering) > 0.20:
                    img_left = load_image(batch_samples.loc[idx, 'left'])
                    img_right = load_image(batch_samples.loc[idx, 'right'])
                    i = add_side_image(x, y, steering, i, batch_size, img_left, img_right)

            yield sklearn.utils.shuffle(x, y)