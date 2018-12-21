import glob
import os
from PIL import Image

count = 1
size = 1 + 1  #第一个数字为视频个数
for count in range(size):
	path = "s:\\test_pic\\tv1\\%d\\*.jpg"%count ###从不同文件夹中读取帧
	img_path = glob.glob(path)
	path_save = "s:\\test_pic\\tv1\\all\\"      ###将裁剪的logo保存到同一文件夹
	for file in img_path:
		name = os.path.join(path_save, file)
		im = Image.open(file)
		im_resize = im.resize((512,288))
		box = [0,0,200,60]
		img = im_resize.crop(box)
		img.save(path_save + name.split('\\')[4],'JPEG')
		count = count + 1
