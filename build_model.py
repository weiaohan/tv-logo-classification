from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D, Activation, Conv2D

def build_model(kHeight, kWidth, input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, (kHeight, kWidth), input_shape=input_shape))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (kHeight, kWidth)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (kHeight, kWidth)))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (kHeight, kWidth)))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (kHeight, kWidth)))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model