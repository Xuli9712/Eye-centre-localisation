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


def generate_target(img, pt, sigma, label_type='Gaussian'):
    if sigma == 0:
        pt_int = (int(pt[1]), int(pt[0])) 
        if 0 <= pt_int[0] < img.shape[0] and 0 <= pt_int[1] < img.shape[1]:
            img[pt_int[0], pt_int[1]] = 1
        return img
    else:
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
                br[0] < 0 or br[1] < 0):
            return img

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2

        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

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

        self.img_size = (img_size, img_size)
        self.heatmap_size = (heatmap_size, heatmap_size)

        self.transformEye = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.333, 0.342], std=[0.209, 0.163, 0.166])
        ])


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.label[idx][0])
            
        pts = self.label[idx][1]
        pupil_dist = self.label[idx][2]
        npoints = len(pts)

        img = Image.open(image_path)
        Height, Width = img.size
        img = self.transformEye(img)

        target = np.zeros((npoints, self.heatmap_size[0], self.heatmap_size[1]))
        tpts = pts.copy()
        eye_pts = []

        for i in range(npoints):
            orig_x, orig_y = tpts[i][0], tpts[i][1]
 
            x_scale = self.heatmap_size[0] / Width
            y_scale = self.heatmap_size[1] / Height

            scaled_x = orig_x * x_scale
            scaled_y = orig_y * y_scale

            target[i] = generate_target(target[i], (scaled_x, scaled_y), self.sigma, label_type='Gaussian')
            eye_pts.append([orig_x, orig_y])
            
        heatmaps = torch.Tensor(target)
        eye_pts = torch.Tensor(eye_pts)
            
        pupil_dist = torch.tensor(pupil_dist)
        Width = torch.tensor(Width)
        Height = torch.tensor(Height)

        
        return img, heatmaps, eye_pts, pupil_dist, Width, Height
    
def extract_keypoints_from_heatmap_to_ori_size(heatmap, width_orig, height_orig):
    num_keypoints = heatmap.shape[0]
    keypoints = np.zeros((num_keypoints, 2))

    for i in range(num_keypoints):
        flat_index = np.argmax(heatmap[i, :, :])  
        y, x = np.unravel_index(flat_index, heatmap[i, :, :].shape)  
        
        x_scale = width_orig / heatmap.shape[2]
        y_scale = height_orig / heatmap.shape[1]
        x = x * x_scale
        y = y * y_scale

        keypoints[i] = [x, y]

    return keypoints

def heatmap2coord(heatmap, org_W, org_H, topk=9):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N, C, 1, -1).topk(topk, dim=-1)
    
    coord = torch.cat([index % W, torch.div(index, W, rounding_mode='trunc')], dim=2)

    keypoints = (coord * F.softmax(score, dim=-1)).sum(-1)

    scale_factor_W = (org_W / W).view(4, 1, 1)
    scale_factor_H = (org_H / H).view(4, 1, 1)
    scale_factors = torch.cat([scale_factor_W, scale_factor_H], dim=-1)
    scale_factors = scale_factors.expand_as(keypoints)
    
    keypoints *= scale_factors

    return keypoints



if __name__ == '__main__':
    #dataTrain = gi4e(data_path=r'F:\gi4e_database\images', csv_path=r'F:\gi4e_database\image_labels.csv',sigma=1.0)
    #img, heatmaps = dataTrain[12]
    dataTrain = gi4e_eye(data_path=r'D:\gi4e_database\blend', json_path=r'D:\gi4e_database\blend_eye_pt.json',sigma=1.0,img_size=(32,32),heatmap_size=(32,32))
    for i in range(50):
        img, heatmaps, eye_pts, pupil_dist, Width, Height = dataTrain[i]
    