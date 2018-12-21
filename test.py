# -*- coding: UTF-8 -*- 
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image,ImageFont,ImageDraw
import build_model
import classfication_dict

model = build_model.build_model(3, 3, (200, 60, 3), 30)
model.load_weights("train_weights/2018-12-21.h5")

img_path = "test_pic/anhui-test.png"
img = image.load_img(img_path, target_size=(200,60))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x, batch_size=1)
result = preds[0]

result_word = classfication_dict.classfication_dict[np.argmax(result, axis=0)]

im = Image.open(img_path)
draw = ImageDraw.Draw(im)
set_font = ImageFont.truetype('/Users/weiaohan/Library/Fonts/Ubuntu Mono derivative Powerline.ttf', 30)
draw.text((100,60), result_word, font=set_font, fill = (255, 0 ,0))
im.show()

print(result)
print(result_word)
