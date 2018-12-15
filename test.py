# -*- coding: UTF-8 -*- 
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import build_model
import classfication_dict

model = build_model.build_model()
model.load_weights("2018-12-15.h5")

img_path = "test_pic/cctv1-test.png"
img = image.load_img(img_path, target_size=(200,60))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x, batch_size=1)
result = preds[0]
print(result)
print(classfication_dict.classfication_dict[np.argmax(result, axis=0)])
