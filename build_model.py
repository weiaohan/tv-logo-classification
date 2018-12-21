from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D, Activation, Conv2D
#残差网络
def build_model(kHeight, kWidth, input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, (kHeight, kWidth), input_shape=input_shape, padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (kHeight, kWidth), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (kHeight, kWidth), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (kHeight, kWidth), padding='same'))#减小卷积核
    model.add(Activation("relu"))
    model.add(Conv2D(128, (kHeight, kWidth), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))#均值池化
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model