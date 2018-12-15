from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D, Activation

def build_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(200, 60, 3)))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(9))
    model.add(Activation("softmax"))
    return model