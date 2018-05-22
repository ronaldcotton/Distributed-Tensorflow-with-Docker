# -*- coding: utf-8 -*-
# from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(units=num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x=x_train, y=y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x=x_test, y=y_test, verbose=2)
import tfDataSet
import keras

modelConfig = { 'datatype': 'image', # or csv
                'reshape x_train': '1D', # TODO: not working atm
                'dataset': 'mnist',
                'epochs': 5,
                'verbose': 2,
                'data_aug': True,
                'cb_CSVLogger': True,
                'appendCSVLogger': False,
                'cb_TensorBoard': True }  # skip model.fit - run ...
 # datagen = ImageDataGenerator, datagen.fit(x_train) , model.fit_generator

imageDataGenerator = { 'featurewise_center': True, 'featurewise_std_normalization': False, 'samplewise_std_normalization': False, 'zca_whitening': True, 'zca_epsilon': 1e-06 }

modelList = [
    { 'Add': { 'function': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'input_shape': 'INPUTSHAPE' } },
    { 'Add': { 'function': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu' } },
    { 'Add': { 'function': 'MaxPooling2D', 'pool_size': (2, 2) } },
    { 'Add': { 'function': 'Dropout', 'rate': 0.25 } },
    { 'Add': { 'function': 'Flatten' } },
    { 'Add': { 'function': 'Dense', 'units': 128, 'activation': 'relu' } },
    { 'Add': { 'function': 'Dropout', 'rate': 0.5 } },
    { 'Add': { 'function': 'Dense', 'units': 'UNITS', 'activation': 'softmax' } },  # num_classes
    { 'Compile': { 'loss': keras.losses.categorical_crossentropy, 'optimizer': keras.optimizers.Adadelta(), 'metrics': ['accuracy'] } },
    #{ 'Fit': { 'x': 'X', 'y': 'Y', 'batch_size': 'BATCHSIZE', 'epochs': 'EPOCHS', 'verbose': 'VERBOSE', 'validation_data': 'VALIDATIONDATA' } }
    { 'datagen.flow': { 'x': 'X', 'y': 'Y', 'batch_size': 'BATCHSIZE' } },
    { 'Fit_generator': { 'epochs': 'EPOCHS', 'verbose': 'VERBOSE', 'validation_data': 'VALIDATIONDATA', 'workers': 4, 'use_multiprocessing': True } }

]
# https://keras.io/callbacks/#earlystopping
# monitor: quantity to be monitored from logs:
#               'val_loss', 'val_acc, 'loss', 'acc'
# min_delta: quits when absolute change is less than min_delta.
# patience: epochs to run after no improvement.
# verbose: verbosity mode.
# mode:
#       min  = training stops when quantity monitored stopped decreasing
#       max  = training stops when quantity monitored stopped increasing
#       auto = direction inferred from the monitored quantity
#                   (accuracy increases, loss decreases)
# if modelStop is not defined, then EarlyStopping callback doesn't exist

# modelStop = { 'monitor': 'val_loss', 'min_delta': 0, 'patience': 0, 'verbose': 0, 'mode': 'auto' }
