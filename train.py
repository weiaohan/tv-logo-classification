# -*- coding: UTF-8 -*- 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D, Activation
from keras import applications
from keras.preprocessing import image
from keras.models import load_model
import time
import matplotlib.pyplot as plt


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

# model.load_weights('first_try.h5')

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rotation_range=40, #随机选择图片的角度
    width_shift_range=0.1,
    height_shift_range=0.1, #指定方向随机移动的成都
    rescale=1./255, #rescale值将在执行其他处理前乘到整个图像上，
                   #我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，
                   #所以我们将这个值定为0~1之间的数
    shear_range=0.1,#剪切变换的程度
    zoom_range=0.2,#放大的程度
    #horizontal_flip=True,#水平翻转
    fill_mode='nearest'#用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(200, 60),
    batch_size=16,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(200, 60),
        batch_size=16,
        class_mode='categorical')

history = model.fit_generator(
   train_generator,
   steps_per_epoch=1000,
   epochs=20,
   validation_data=validation_generator,
   validation_steps=400)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
model.save_weights(time_stamp + '.h5')
