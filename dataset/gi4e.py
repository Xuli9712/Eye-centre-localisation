import os,sys
import random
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import json
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到Python的模块搜索路径中
sys.path.append(parent_dir)


class gi4e(data.Dataset):
    def __init__(self, data_path, csv_path, sigma, img_size=(256, 256), heatmap_size=(64, 64)):
        self.data_path = data_path
        self.landmarks_frame = pd.read_csv(csv_path)
        self.sigma = sigma

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.img_size = img_size
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.landmarks_frame.iloc[idx, 0])
        
        pts = self.landmarks_frame.iloc[idx, 1:13].values
        pts = pts.astype('float').reshape(-1, 2)
        npoints = pts.shape[0]

        # load image and convert to rgb
        img = Image.open(image_path).convert('RGB')
        H, W = img.size
        ori_img = img.copy()

        img = np.array(img, dtype=np.float32)
        target = np.zeros((npoints, self.heatmap_size[0], self.heatmap_size[1]))
        tpts = pts.copy()

        for i in range(npoints):
            if tpts[i, 1] > 0:  # 如果关键点的坐标有效
                # 首先获取关键点在原图中的坐标
                orig_x, orig_y = tpts[i, 0], tpts[i, 1]

                # 计算缩放比例
                x_scale = self.heatmap_size[0] / W
                y_scale = self.heatmap_size[1] / H

                # 应用缩放
                scaled_x = orig_x * x_scale
                scaled_y = orig_y * y_scale

                # 使用缩放后的坐标生成热力图
                target[i] = generate_target(target[i], (scaled_x, scaled_y), self.sigma, label_type='Gaussian')

        # resize image and points
        # 将图像转换为PIL Image对象，并调整其尺寸
        img = F.resize(Image.fromarray(img.astype(np.uint8)), self.img_size)

        # 将PIL Image对象转换回numpy ndarray，并转换数据类型
        img = np.array(img).astype(np.float32)

        # 标准化图像并调整维度顺序
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)

        heatmaps = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        '''
        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}
        convert_pts = extract_keypoints_from_heatmap(target, W, H)
        
        # Display original and resized images with landmarks
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 7, 1)
        plt.imshow(ori_img)
        plt.scatter(convert_pts[:, 0], convert_pts[:, 1], s=5, marker='.', c='r')
        plt.title("Original Image")

        for i in range(npoints):
            plt.subplot(1, 8, i+3)
            plt.imshow(target[i], cmap='hot')
            plt.title(f"Target {i+1}")

        plt.show()
        '''
       
        
        # 返回的图像为256*256大小，热力图为64*64 pts为在原图尺寸上的坐标位置

        return img, heatmaps


def generate_target(img, pt, sigma, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

class gi4e_eye(data.Dataset):
    def __init__(self, data_path, json_path, sigma, img_size, heatmap_size):
        self.data_path = data_path
        with open(json_path) as f:
            self.label = json.load(f)
        
        self.sigma = sigma

        self.img_size = img_size
        self.heatmap_size = heatmap_size

        self.transformEye = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.333, 0.342], std=[0.209, 0.163, 0.166])
        ])


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.data_path, self.label[idx][0])
            
            pts = self.label[idx][1]
            npoints = len(pts)

            img = Image.open(image_path)
            ori_img = img.copy()
            H, W = img.size
            

            img = self.transformEye(img)

            target = np.zeros((npoints, self.heatmap_size[0], self.heatmap_size[1]))
            tpts = pts.copy()

            for i in range(npoints):
                orig_x, orig_y = tpts[i][0], tpts[i][1]
 
                x_scale = self.heatmap_size[0] / W
                y_scale = self.heatmap_size[1] / H

                scaled_x = orig_x * x_scale
                scaled_y = orig_y * y_scale

                target[i] = generate_target(target[i], (scaled_x, scaled_y), self.sigma, label_type='Gaussian')

            heatmaps = torch.Tensor(target)
            tpts = torch.Tensor(tpts)
            '''
            convert_pts = extract_keypoints_from_heatmap_to_ori_size(target, W, H)
            
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 7, 1)
            plt.imshow(ori_img)
            plt.scatter(convert_pts[:, 0], convert_pts[:, 1], s=5, marker='.', c='r')
            plt.title("Original Image")
            for i in range(npoints):
                plt.subplot(1, 8, i+3)
                plt.imshow(target[i], cmap='hot')
                plt.title(f"Target {i+1}")
            plt.show()
            '''
            
        except Exception as e:
            idx = idx-1
            image_path = os.path.join(self.data_path, self.label[idx][0])
            
            pts = self.label[idx][1]
            npoints = len(pts)

            # load image and convert to rgb
            img = Image.open(image_path)
            
            ori_img = img.copy()
            H, W = img.size


            img = self.transformEye(img)

            target = np.zeros((npoints, self.heatmap_size[0], self.heatmap_size[1]))
            tpts = pts.copy()

            for i in range(npoints):
                orig_x, orig_y = tpts[i][0], tpts[i][1]
 
                x_scale = self.heatmap_size[0] / W
                y_scale = self.heatmap_size[1] / H

                scaled_x = orig_x * x_scale
                scaled_y = orig_y * y_scale

                target[i] = generate_target(target[i], (scaled_x, scaled_y), self.sigma, label_type='Gaussian')


            heatmaps = torch.Tensor(target)
            tpts = torch.Tensor(tpts)
            
            meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}
            convert_pts = extract_keypoints_from_heatmap_to_ori_size(target, W, H)
            '''
           # Display original and resized images with landmarks
            plt.figure(figsize=(10, 10))

            plt.subplot(1, npoints+2, 1)
            plt.imshow(ori_img)
            plt.scatter(convert_pts[:, 0], convert_pts[:, 1], s=5, marker='.', c='r')
            plt.title("Original Eye Image")

            for i in range(npoints):
                plt.subplot(1, npoints+2, i+2)  
                plt.imshow(target[i], cmap='hot')
                plt.title(f"Target Heatmap{i+1}")

            plt.show()
            '''

            
        return img, heatmaps

def extract_keypoints_from_heatmap_to_ori_size(heatmap, width_orig, height_orig):
    num_keypoints = heatmap.shape[0]
    keypoints = np.zeros((num_keypoints, 2))

    for i in range(num_keypoints):
        flat_index = np.argmax(heatmap[i, :, :])  # 找到最大值的一维索引
        y, x = np.unravel_index(flat_index, heatmap[i, :, :].shape)  # 将一维索引转为二维坐标
        
        # 转换坐标到原图的尺寸
        x_scale = width_orig / heatmap.shape[2]
        y_scale = height_orig / heatmap.shape[1]
        x = x * x_scale
        y = y * y_scale

        keypoints[i] = [x, y]

    return keypoints


if __name__ == '__main__':
    #dataTrain = gi4e(data_path=r'F:\gi4e_database\images', csv_path=r'F:\gi4e_database\image_labels.csv',sigma=1.0)
    #img, heatmaps = dataTrain[12]
    dataTrain = gi4e_eye(data_path=r'F:\gi4e_database\blend', json_path=r'F:\gi4e_database\blend_eye_pt.json',sigma=1.0,img_size=(64, 64),heatmap_size=(64, 64))
    for i in range(50):
        img, heatmaps = dataTrain[i]
    