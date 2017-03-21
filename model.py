import argparse
import os, json

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten, Dropout, SpatialDropout2D, Cropping2D, Convolution2D, Lambda
from keras.models import Sequential
from keras.optimizers import Nadam, adam

from data_io import generator, get_data

#
#[Reference]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
#

def nvidia_dropout_model(input_shape=(80, 160, 3)):

    ch, row, col = 3, 80, 320  # Trimmed image format
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=input_shape))
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation="relu", border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="relu", border_mode='same'))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

def train(model_path='model.h5'):
    epochs = 5
    batch_size = 64
    input_shape = (160, 320, 3)

    m = nvidia_dropout_model(input_shape=input_shape)

    optimizer = Nadam()
    m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[])

    train_samples, validation_samples, samples_per_epoch_training, samples_per_epoch_validation = get_data(batch_size, epochs)
    print('Training size: %d samples per epoch: %d, Validation size: %d samples per epoch: %d'%(len(train_samples), samples_per_epoch_training, len(validation_samples), samples_per_epoch_validation))
    print("------------------------------------------------------------------------------------")

    checkpointer = ModelCheckpoint(filepath=os.path.join(os.path.split(__file__)[0], model_path),
                                   verbose=1, save_best_only=True)

    train_generator = generator(train_samples, batch_size=batch_size, input_shape=input_shape)
    validation_generator = generator(validation_samples, batch_size=batch_size,
                                                             input_shape=input_shape)

    history = m.fit_generator(train_generator,
                              samples_per_epoch=samples_per_epoch_training, nb_epoch=epochs, verbose=1,
                              validation_data=validation_generator,
                              nb_val_samples=samples_per_epoch_validation, pickle_safe=True,
                              callbacks=[checkpointer])

    score = m.evaluate_generator(validation_generator,
                                 val_samples=samples_per_epoch_validation, pickle_safe=True)

    print('Validation MSE:', score)
    return m, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5 to be saved')
    args = parser.parse_args()

    model, history = train(args.model)

    model_rep = model.to_json()

    # Save data
    with open('model.json', 'w') as f:
        json.dump(model_rep, f)

    model.save('./'+args.model+'_final.h5')