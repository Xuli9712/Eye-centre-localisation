import cv2
import os, sys
from pathlib import Path
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch

from tools.mediapipe_eye_detector import MPFacialLandmarkDetector
from model.Unet import UNet
from model.Unetpp import NestedUNet
from tools.heatmap_trans import *


class Preprocessor:
    def __init__(self, img_size, heatmap_size, landmark_detector):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.landmark_detector = landmark_detector
        self.transformEye = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.333, 0.342], std=[0.209, 0.163, 0.166])
        ])

    def preprocess_frame_to_tensor(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            landmarks = self.landmark_detector.get_landmarks(img)

            if landmarks is not None:
                for landmark in landmarks:
                    self.landmark = landmark
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157]
                    right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
                        
                    left_eye_points = [landmark[i] for i in left_eye_indices]
                    right_eye_points = [landmark[i] for i in right_eye_indices]

                    left_eye, Leye_bbox_pts= self.landmark_detector.get_square_roi(img, left_eye_points, scale=1.5)
                    right_eye, Reye_bbox_pts = self.landmark_detector.get_square_roi(img, right_eye_points, scale=1.5)
                    Leye_img = Image.fromarray(left_eye)
                    Reye_img = Image.fromarray(right_eye)
                    left_w, left_h = Leye_img.size
                    right_w, right_h = Reye_img.size
                    Leye = self.transformEye(Leye_img).unsqueeze(0)
                    Reye = self.transformEye(Reye_img).unsqueeze(0)
            return Leye, Reye, Leye_bbox_pts, Reye_bbox_pts, left_w, left_h, right_w, right_h
    
        except Exception as e:
            print(e)
        
        
def convert_keypoints_to_original_frame(left_eye_heatmap_output, right_eye_heatmap_output, Leye_bbox_pts, Reye_bbox_pts, left_w, left_h, right_w, right_h):
    # 先将热力图映射回裁剪出来眼睛图片的尺寸输出keypoints
    left_w = torch.tensor([left_w]).unsqueeze(0).cuda()
    left_h = torch.tensor([left_h]).unsqueeze(0).cuda()
    right_w = torch.tensor([right_w]).unsqueeze(0).cuda()
    right_h = torch.tensor([right_h]).unsqueeze(0).cuda()
    left_keypoints = extract_keypoints_from_heatmap_to_ori_size_batch(left_eye_heatmap_output, left_w, left_h).squeeze(0).cpu()
    right_keypoints = extract_keypoints_from_heatmap_to_ori_size_batch(right_eye_heatmap_output, right_w, right_h).squeeze(0).cpu()
    # 裁剪出的小图的左上角坐标为 左眼：(Leye_bbox_pts[0][0], Leye_bbox_pts[0][1]) 右眼(Reye_bbox_pts[0][0], Reye_bbox_pts[0][1]) 只要每个点加上这个点即可
    left_keypoints = np.array(left_keypoints) + np.array([Leye_bbox_pts[0][0], Leye_bbox_pts[0][1]])
    right_keypoints = np.array(right_keypoints) + np.array([Reye_bbox_pts[0][0], Reye_bbox_pts[0][1]])
    return left_keypoints, right_keypoints

def load_model(model, weight_path):
    if weight_path is None:
        print("No weights path provided!")
        sys.exit(1)

    try:
        pretrained_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("Weights {} loaded successfully.".format(os.path.basename(weight_path)))
    except FileNotFoundError:
        print("Weights file not found!")
        sys.exit(1)
    except Exception as e:
        print("Error loading pretrained weights:", e)
        sys.exit(1)
    return model

def main(img_size=(64,64), heatmap_size=(64,64), weight_path = r'C:\Users\Xuli\Desktop\hrnet_gi4e\Best\best.pth'):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # facail landmark 检测器
    landmark_detector = MPFacialLandmarkDetector()
    # 原图->送入模型的左右眼tensor 预处理器
    preprocessor = Preprocessor(img_size, heatmap_size, landmark_detector=landmark_detector)
    model = NestedUNet()
    model.eval()
    model = load_model(model, weight_path)
    model = model.cuda()
    # 使用OpenCV打开摄像头
    cap = cv2.VideoCapture(0)

    # 循环获取摄像头的画面并进行推理
    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        if not ret:
            break
        start = time.time()
        # 将图像转化为模型需要的输入格式，比如转为Tensor，做归一化等操作
        Leye, Reye, Leye_bbox_pts, Reye_bbox_pts, left_w, left_h, right_w, right_h = preprocessor.preprocess_frame_to_tensor(frame)
        with torch.no_grad():
            Leye = Leye.cuda()
            Reye = Reye.cuda()
            left_eye_heatmap_output = model(Leye)
            right_eye_heatmap_output = model(Reye)
        left_eye_heatmap_output = left_eye_heatmap_output      #.squeeze().cpu().numpy()
        right_eye_heatmap_output = right_eye_heatmap_output     #.squeeze().cpu().numpy()
            # 输出转换为原图的坐标点
        left_keypoints, right_keypoints = convert_keypoints_to_original_frame(left_eye_heatmap_output, right_eye_heatmap_output, Leye_bbox_pts, Reye_bbox_pts, left_w, left_h, right_w, right_h)
            # 将点绘制在原图上
            # 左眼
        for i in range(3):
            cv2.circle(frame, tuple(map(int, left_keypoints[i])), 1, (0, 0, 255), -1)
            # 右眼
        for j in range(3):
            cv2.circle(frame, tuple(map(int, right_keypoints[j])), 1, (0, 0, 255), -1)
        print("Process Time(s) Per Image", time.time()-start)
        cv2.imshow('Image', frame)
        cv2.waitKey(100)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(img_size=(64,64), heatmap_size=(64,64), weight_path = r'C:\Users\Xuli\Desktop\hrnet_gi4e\Best\best.pth')

            
 


    








