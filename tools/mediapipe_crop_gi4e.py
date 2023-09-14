from mediapipe_eye_detector import MPFacialLandmarkDetector
import cv2
import os
import concurrent.futures
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import json
import pandas as pd
import matplotlib.pyplot as plt
import math


def process_images(img_path, img_name, pts, landmark_detector, root_path, eye_label):
    blend_eye_path = os.path.join(root_path, 'blend')
    if not os.path.exists(blend_eye_path):
            os.makedirs(blend_eye_path)
    Leye_img_path = os.path.join(root_path, 'Leyes')
    if not os.path.exists(Leye_img_path):
            os.makedirs(Leye_img_path)
    Reye_img_path = os.path.join(root_path, 'Reyes')
    if not os.path.exists(Reye_img_path):
            os.makedirs(Reye_img_path) 
   
    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    landmarks = landmark_detector.get_landmarks(img)

    if landmarks is not None:
        for landmark in landmarks:
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157]
            right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]


            left_eye_points = [landmark[i] for i in left_eye_indices]
            right_eye_points = [landmark[i] for i in right_eye_indices]

                #face = landmark_detector.get_square_roi(img, landmark, scale=1.2)
            face, face_bbox_pts = landmark_detector.get_square_roi(img, landmark, scale=1.2)
            left_eye, Leye_bbox_pts= landmark_detector.get_square_roi(img, left_eye_points, scale=1.5)
            right_eye, Reye_bbox_pts = landmark_detector.get_square_roi(img, right_eye_points, scale=1.5)

            # convert the eye points in the original image to the cropped eye image
            left_eye_pts = [[pt[0] - Leye_bbox_pts[0][0], pt[1] - Leye_bbox_pts[0][1]] for pt in pts[3:]]
            right_eye_pts = [[pt[0] - Reye_bbox_pts[0][0], pt[1] - Reye_bbox_pts[0][1]] for pt in pts[:3]]

            # 计算pupillary distance 用于normalize error
            pupil_dis = round(math.sqrt((pts[1][0] - pts[4][0])**2 + (pts[1][1] - pts[4][1])**2),5)

            eye_pts = [left_eye_pts, right_eye_pts]
            #eye_label[img_name] = eye_pts
            l_img_name = 'L_' + img_name 
            r_img_name = 'R_' + img_name

            eye_label.append([l_img_name, left_eye_pts, pupil_dis])
            eye_label.append([r_img_name, right_eye_pts, pupil_dis])
            '''
            # draw the points on the cropped eye images
            for pt in left_eye_pts:
                cv2.circle(left_eye, tuple(map(int, pt)), 2, (0, 0, 255), -1)
            for pt in right_eye_pts:
                cv2.circle(right_eye, tuple(map(int, pt)), 2, (0, 0, 255), -1)
            '''
            cv2.imwrite(os.path.join(blend_eye_path, l_img_name), left_eye)
            cv2.imwrite(os.path.join(blend_eye_path, r_img_name), right_eye)

            '''
            # show the images with landmarks
            plt.figure(figsize=(10,5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB))
            plt.title('Left Eye')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB))
            plt.title('Right Eye')
            plt.show()
            '''
def process_dataset(root_path):
    landmark_detector = MPFacialLandmarkDetector()
    img_root = os.path.join(root_path, 'images')
    csv_path = os.path.join(root_path, 'image_labels.csv')
    landmarks_frame = pd.read_csv(csv_path)
    eye_label = []
    for i in range(len(landmarks_frame)):
        img_name = landmarks_frame.iloc[i, 0]
        img_path = os.path.join(img_root, img_name)
        pts = landmarks_frame.iloc[i, 1:13].values
        pts = pts.astype('float').reshape(-1, 2)
        process_images(img_path, img_name, pts, landmark_detector, root_path, eye_label)
    with open('blend_eye_pt.json','w') as f:
        json.dump(eye_label, f)


if __name__ == '__main__':
    root_path = r'D:\gi4e_database'
    process_dataset(root_path)
    
