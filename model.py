import argparse
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Convolution2D, Flatten, Dropout, SpatialDropout2D, Cropping2D
from keras.models import Sequential
from keras.optimizers import Nadam

from data_io import generate_train, generate_valid, get_data

#
#[Reference]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
#

def nvidia_dropout_model(input_shape=(80, 160, 3)):

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

def train(model_path='model.h5'):
    epochs = 3
    batch_size = 64
    input_shape = (160, 320, 3)

    m = nvidia_dropout_model(input_shape=input_shape)

    optimizer = Nadam()
    m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[])

    train_samples, validation_samples, samples_per_epoch_training, samples_per_epoch_validation = get_data(batch_size)
    print('Training size: %d, Validation size: %d'%(len(train_samples), len(validation_samples)))
    print("----------------------------------------------------")

    checkpointer = ModelCheckpoint(filepath=os.path.join(os.path.split(__file__)[0], model_path),
                                   verbose=1, save_best_only=True)

    history = m.fit_generator(generate_train(train_samples, batch_size=batch_size, input_shape=input_shape),
                              samples_per_epoch=samples_per_epoch_training, nb_epoch=epochs, verbose=1,
                              validation_data=generate_valid(validation_samples, batch_size=batch_size,
                                                             input_shape=input_shape),
                              nb_val_samples=samples_per_epoch_validation, pickle_safe=True,
                              callbacks=[checkpointer])

    score = m.evaluate_generator(generate_valid(validation_samples, batch_size=batch_size, input_shape=input_shape),
                                 val_samples=samples_per_epoch_validation, pickle_safe=True)
    print('Validation MSE:', score)

    return m, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5 to be saved')
    args = parser.parse_args()

    train(args.model)