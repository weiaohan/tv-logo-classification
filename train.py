# -*- coding: UTF-8 -*- 
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import build_model

epochs = 25
steps_per_epoch = 1250
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument('-p', default=False, type=bool, help='plot loss and accuracy')
args = parser.parse_args()

model = build_model.build_model(3, 3, (150, 150, 3), 30)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
    rescale=1./255, #rescale值将在执行其他处理前乘到整个图像上，
                   #我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，
                   #所以我们将这个值定为0~1之间的数
    #horizontal_flip=True,#水平翻转
    fill_mode='constant',#用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
    validation_split=0.1,
    cval=0
)
train_generator = datagen.flow_from_directory(
    'data',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
    )#查看分类混合结果
validation_generator = datagen.flow_from_directory(
        'data',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

history = model.fit_generator(
   train_generator,
   steps_per_epoch=steps_per_epoch,
   epochs=epochs,
   validation_data=validation_generator
   )

if args.p:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
model.save_weights(time_stamp + '.h5')
