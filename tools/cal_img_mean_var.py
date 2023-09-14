import numpy as np
import os
import cv2
import random
from tqdm import tqdm
import json
# calculate means and std
data_path = r'F:\gi4e_database\blend'
json_path = r'F:\gi4e_database\blend_eye_pt.json'

img_h, img_w = 64, 64
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
with open(json_path, 'r') as f:
   meta = json.load(f)
   for item in tqdm(meta):
      imgname = item[0]
      img_path = os.path.join(data_path, imgname)
      img = cv2.imread(img_path)
      img = cv2.resize(img, (img_h, img_w))
      img = img[:, :, :, np.newaxis]
      imgs = np.concatenate((imgs, img), axis=3)
      # print(i)
imgs = imgs.astype(np.float32)/255.
for i in tqdm(range(3)):
      pixels = imgs[:,:,i,:].ravel() # 拉成一行
      means.append(np.mean(pixels))
      stdevs.append(np.std(pixels))
# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse() # BGR --> RGB
stdevs.reverse()
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))