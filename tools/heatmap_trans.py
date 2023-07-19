import numpy as np

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

