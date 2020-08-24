from keras.callbacks import EarlyStopping
from keras.engine.saving import model_from_json
from keras.layers import Activation, add, Concatenate, Input, Dense, BatchNormalization, Conv1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import os
from keras import Model
from compress_pickle import load


def convtimenet_block2(x_shape):
    input_layer = Input((None, x_shape[2]))
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5])
    batchnorm = BatchNormalization()(concatted)
    return Model(input_layer, batchnorm)


def convtimenet_block1(x_shape):
    input_layer = Input((None, x_shape[2]))
    x = convtimenet_block2(x_shape)(input_layer)
    x = Activation(activation="relu")(x)
    return Model(input_layer, x)


def convtimenet_full_layer(x_shape):
    input_layer = Input((None, x_shape[2]))
    x = convtimenet_block1(x_shape)(input_layer)
    x = convtimenet_block2(x.get_shape().as_list())(x)
    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer)
    x = add([x, conv1])
    x = Activation('relu')(x)
    return Model(input_layer, x)


def bigconvtimenet_6(x_shape, num_tasks):
    # Input layer
    input_layer = Input((None, x_shape[2]))

    # First layer
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    concatted2 = Concatenate()([convolution11, convolution21, convolution31, convolution41, convolution51])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    # Second layer
    input_layer2 = x
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer2)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer2)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer2)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer2)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer2)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    concatted2 = Concatenate()([convolution11, convolution21, convolution31, convolution41, convolution51])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer2)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    # Third layer
    input_layer3 = x
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer3)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer3)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer3)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer3)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer3)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    concatted2 = Concatenate()([convolution11, convolution21, convolution31, convolution41, convolution51])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer3)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    # Forth layer
    input_layer4 = x
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer4)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer4)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer4)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer4)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer4)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    concatted2 = Concatenate()([convolution11, convolution21, convolution31, convolution41, convolution51])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer4)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    input_layer5 = x
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer5)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer5)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer5)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer5)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer5)
    convolution6 = Conv1D(filters=33, kernel_size=128, padding='same')(input_layer5)

    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5, convolution6])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    convolution61 = Conv1D(filters=33, kernel_size=128, padding='same')(x)
    concatted2 = Concatenate()(
        [convolution11, convolution21, convolution31, convolution41, convolution51, convolution61])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer5)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    input_layer6 = x
    convolution1 = Conv1D(filters=33, kernel_size=4, padding='same')(input_layer6)
    convolution2 = Conv1D(filters=33, kernel_size=8, padding='same')(input_layer6)
    convolution3 = Conv1D(filters=33, kernel_size=16, padding='same')(input_layer6)
    convolution4 = Conv1D(filters=33, kernel_size=32, padding='same')(input_layer6)
    convolution5 = Conv1D(filters=33, kernel_size=64, padding='same')(input_layer6)
    convolution6 = Conv1D(filters=33, kernel_size=128, padding='same')(input_layer6)
    concatted = Concatenate()([convolution1, convolution2, convolution3, convolution4, convolution5, convolution6])
    batchnorm = BatchNormalization()(concatted)
    x = Activation(activation="relu")(batchnorm)

    convolution11 = Conv1D(filters=33, kernel_size=4, padding='same')(x)
    convolution21 = Conv1D(filters=33, kernel_size=8, padding='same')(x)
    convolution31 = Conv1D(filters=33, kernel_size=16, padding='same')(x)
    convolution41 = Conv1D(filters=33, kernel_size=32, padding='same')(x)
    convolution51 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    convolution61 = Conv1D(filters=33, kernel_size=64, padding='same')(x)
    concatted2 = Concatenate()(
        [convolution11, convolution21, convolution31, convolution41, convolution51, convolution61])
    batchnorm2 = BatchNormalization()(concatted2)

    conv1 = Conv1D(filters=1, kernel_size=1)(input_layer6)
    x = add([batchnorm2, conv1])
    x = Activation('relu')(x)

    gap_layer = GlobalAveragePooling1D()(x)
    output = Dense(num_tasks, activation="linear")(gap_layer)
    model = Model(input_layer, output)
    model.compile(loss='MSE',
                  optimizer="adam",
                  metrics=['mse'])
    return model


def find_max_files(check_path):
    max_len = 0
    for file in os.listdir(check_path):
        files = list(os.listdir(f'{check_path}{file}/'))
        if len(files) < 1:
            continue
        max_len = 0
        part = max([int(file2.split("_")[1].split("part")[1]) for file2 in files])
        if part > max_len:
            max_len = part
    return max_len


def find_number_of_files(check_path):
    max_len = 0
    for file in os.listdir(check_path):
        current_len = len(os.listdir(f'{check_path}{file}/'))
        if current_len > max_len:
            max_len = current_len
    return int(max_len / 2) - 1


def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model


def defreeze_layers(model):
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='MSE',
                  optimizer="adam",
                  metrics=['mse'])
    return model


def find_latest(path):
    max_file = 0
    for file in os.listdir(path):
        spl = file.split("_")[1]
        number = int(spl.split(".")[0])
        if number > max_file:
            max_file = number
    return max_file


def load_model(model_path, start_counter):
    last_model = start_counter
    json_file = open(model_path + 'model_{}.json'.format(last_model), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path + "weight_{}.h5".format(last_model))
    for layer in loaded_model.layers:
        layer.trainable = True
    loaded_model.compile(loss='MSE',
                  optimizer="adam",
                  metrics=['mse'])
    return loaded_model


def train_model(data_path, model_path, max_sequence_length=8001, model="convtime_6",
                save_each_file=True, printing=True, start_counter=0):
    """training a model on synthetic data.

        Parameters:
        data_path : str.
            Path to the where the synthetic generated data was saved.

        model_path : str.
            Path to where the model will be saved

        max_sequence_length : int, default = 8001
            For memory reasons, not all computers can load the data above sequence length. if this is the case for you
            just check what is the maximum length your computer can load.
            default is 8001, which means don't skip any sequence length (max sequence length at generated data is 8000)

        model : str or Keras model, default = "convtime_6"
            the model we want to train on our synthetic data. if using default we load our "ConvTime6" archivtecture.
            Feel free to provide your own model, but remember that last layer should match the y (target task).
            if you use our own generated data, than last layer should be 55 linear actiovation function.

        printing : bool, default = True
            If you want to track the model's learning process (not verbose - just files that was trained).

        start_counter : int, default = 0
            If code crashes, we don't want to train the model from scratch. so we can go to model directory, and check
            what was our last checkpoint.
            If start_counter > 0, model will load automaticly the model at checkpoint for start_counter.

        Returns
        -------
        None, cause all files will be written to disk.

       """
    number_of_parts = find_max_files(data_path)
    counter = 0
    if start_counter > 0:
        model = load_model(model_path, start_counter)
    for i in range(number_of_parts):
        if printing:
            print(f"Currently on part {i+1}/{number_of_parts}")
        for file in os.listdir(data_path):
            if int(file) > max_sequence_length:
                continue
            counter += 1
            if counter <= start_counter:
                continue
            if printing:
                print(f"Training sequence length: {file}")
            es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
            path = f"{data_path}{file}/{file}_part"
            x_train = load(f'{path}{i}_x_test.gz')[:100, :]
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            Y_train = load(f'{path}{i}_y_test.gz')[:100, :]
            X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, test_size=0.2, random_state=41,
                                                                shuffle=True)
            if model == "convtime_6":
                model = bigconvtimenet_6(X_train.shape, y_train.shape[1])
            model = defreeze_layers(model)
            model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=128, callbacks=[es],
                      validation_data=(X_test, y_test))
            if save_each_file:
                model2 = freeze_layers(model)
                model2.save_weights(f'{model_path}/weight_{counter}.h5')
                model_json = model2.to_json()
                with open(f'{model_path}model_{counter}.json', "w") as json_file:
                    json_file.write(model_json)
                json_file.close()
            if counter == 4:
                print("...")
                print("and so on for all parts and all sequence length")
                break
        if counter == 4:
            break
    model2 = freeze_layers(model)
    model2.save_weights(f'{model_path}/weight_final{counter}.h5')
    model_json = model2.to_json()
    with open(f'{model_path}model_final{counter}.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()




