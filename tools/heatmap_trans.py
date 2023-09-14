import numpy as np
import torch
import torch.nn.functional as F

def extract_keypoints_from_heatmap(heatmap):
    num_keypoints = heatmap.shape[0]
    keypoints = np.zeros((num_keypoints, 2))

    for i in range(num_keypoints):
        flat_index = np.argmax(heatmap[i, :, :])  # 找到最大值的一维索引
        y, x = np.unravel_index(flat_index, heatmap[i, :, :].shape)  # 将一维索引转为二维坐标
        keypoints[i] = [x, y]

    return keypoints

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


def soft_argmax2d(heatmap):
    # 获取坐标网格
    x_range = np.arange(heatmap.shape[1])
    y_range = np.arange(heatmap.shape[0])
    X, Y = np.meshgrid(x_range, y_range)
    
    # 计算加权平均坐标
    x_coord = np.sum(X * heatmap) / np.sum(heatmap)
    y_coord = np.sum(Y * heatmap) / np.sum(heatmap)
    
    return x_coord, y_coord

def soft_extract_keypoints_from_heatmap(heatmap, width_orig, height_orig):
    num_keypoints = heatmap.shape[0] # 获取关键点的数量
    keypoints = np.zeros((num_keypoints, 2)) # 初始化关键点坐标数组

    for i in range(num_keypoints):
        # 使用soft-argmax提取关键点坐标
        x, y = soft_argmax2d(heatmap[i, :, :])

        # 计算缩放比例
        x_scale = width_orig / heatmap.shape[2]
        y_scale = height_orig / heatmap.shape[1]

        # 将关键点坐标映射回原图的尺寸
        x = x * x_scale
        y = y * y_scale

        # 保存关键点坐标
        keypoints[i] = [x, y]

    return keypoints


def extract_keypoints_from_heatmap_to_ori_size_batch(heatmaps, widths_orig, heights_orig):
    batch_size, num_keypoints, _, _ = heatmaps.shape
    keypoints = torch.zeros((batch_size, num_keypoints, 2), device=heatmaps.device)

    for b in range(batch_size):
        for i in range(num_keypoints):
            flat_index = torch.argmax(heatmaps[b, i, :, :])  # 找到最大值的一维索引
            y, x = torch.div(flat_index, heatmaps.shape[3], rounding_mode='trunc'), flat_index % heatmaps.shape[3]  # 将一维索引转为二维坐标

            # 转换坐标到原图的尺寸
            x_scale = widths_orig[b] / heatmaps.shape[3]
            y_scale = heights_orig[b] / heatmaps.shape[2]
            x = x.float() * x_scale
            y = y.float() * y_scale

            keypoints[b, i] = torch.tensor([x, y], device=heatmaps.device)

    return keypoints

# 输入均为放cuda上的一个batch的tensor数据，返回该batch根据映射回原切出眼睛图像的坐标系上的keypoints坐标与label的欧式距离，除以瞳孔距离
def cal_batch_error(pred_hm, eye_pts, Width, Height, pupil_dist, randomrounding):
    #pred_pts = extract_keypoints_from_heatmap_to_ori_size_batch(pred_hm, Width, Height)
    if randomrounding == 1:
        pred_pts = heatmap2coord(pred_hm, Width, Height, topk=9)
    else:
        pred_pts = extract_keypoints_from_heatmap_to_ori_size_batch(pred_hm, Width, Height)
    eu_dist = torch.sqrt(torch.sum((pred_pts - eye_pts)**2, dim=-1))
    norm_eu_dist = eu_dist / pupil_dist
    error = torch.mean(eu_dist)
    norm_error = torch.mean(norm_eu_dist)
    return error, norm_error, norm_eu_dist


    

# 使用random rounding 转换heatmaps->keypoints 注意，输入输出都是cuda上的tensor
def heatmap2coord(heatmap, org_W, org_H, topk=9):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N, C, 1, -1).topk(topk, dim=-1)
    
    # Here we replace 'index // W' with 'torch.div(index, W, rounding_mode='trunc')'
    coord = torch.cat([index % W, torch.div(index, W, rounding_mode='trunc')], dim=2)

    # calculate the coordinates in original size
    keypoints = (coord * F.softmax(score, dim=-1)).sum(-1)

    # scale the coordinates back to original image size
    # 假设你的keypoints的形状是(N,3,2)，两个缩放因子的形状都是(N,1,1)
    scale_factor_W = (org_W / W).view(N, 1, 1)
    scale_factor_H = (org_H / H).view(N, 1, 1)
    scale_factors = torch.cat([scale_factor_W, scale_factor_H], dim=-1)
    scale_factors = scale_factors.expand_as(keypoints)
    
    keypoints *= scale_factors

    return keypoints


