import cv2
import mediapipe as mp
import numpy as np

class MPFacialLandmarkDetector:
    def __init__(self, static_image_mode=True):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode)

    def get_landmarks(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                landmarks.append([(int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])) for landmark in face_landmarks.landmark])
            return landmarks
        else:
            return None

    def get_square_roi(self, img, points, scale):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
    
        x1 = max(0, min(x_coords))
        x2 = min(img.shape[1], max(x_coords))
        y1 = max(0, min(y_coords))
        y2 = min(img.shape[0], max(y_coords))
    
        width = x2 - x1
        height = y2 - y1
    
        width_diff = int(width * (scale - 1) / 2)
        height_diff = int(height * (scale - 1) / 2)
    
        x1 = max(0, x1 - width_diff)
        x2 = min(img.shape[1], x2 + width_diff)
        y1 = max(0, y1 - height_diff)
        y2 = min(img.shape[0], y2 + height_diff)
    
        width = x2 - x1
        height = y2 - y1
    
        if width > height:
            diff = (width - height) // 2
            y1 = max(0, y1 - diff)
            y2 = min(img.shape[0], y2 + diff)
        else:
            diff = (height - width) // 2
            x1 = max(0, x1 - diff)
            x2 = min(img.shape[1], x2 + diff)
    
        return img[y1:y2, x1:x2], [(x1,y1),(x2,y2)]
    
    
        



