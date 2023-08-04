import cv2
import torch
import numpy as np
import sys
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import functional as F
from model.Unet import UNet
from model.Unetpp import NestedUNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import json
from tools.heatmap_trans import *
from tqdm import tqdm

def predict_eye_center(model, img_path, weight_path, img_size):
    # eval mode
    
    # Load and preprocess the image
    image = Image.open(img_path)
    img_copy = image.copy()
    W, H = img_copy.size
    #img_copy = img_copy.resize(img_size)
    W = torch.tensor(W).cuda()
    H = torch.tensor(H).cuda()
    
    image = preprocess_image(image, img_size)
    
    image = image.cuda()

    with torch.no_grad():
        output = model(image)
    
    num_keypoints = 3

    keypoints = heatmap2coord(output, W, H)
    keypoints = keypoints.squeeze().cpu().numpy()
    
    #print(keypoints)

    for i in range(num_keypoints):
        img_copy = cv2.circle(np.array(img_copy), tuple(map(int, keypoints[i])), 1, (0, 0, 255), -1)

    output = output.squeeze().cpu().numpy()
    '''
    # 显示左侧的单张图像, 点已经绘制上去了
    plt.subplot(1, 5, 1)
    plt.imshow(img_copy)

    # 显示右侧的热力图
    for i in range(3):
        plt.subplot(1, 5, i+3)
        plt.imshow(output[i], cmap='hot')
        plt.title(f"Target {i+1}")

    plt.tight_layout()  
    plt.show()
    '''

    return keypoints

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


def preprocess_image(image, img_size):
    transformEye = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.591, 0.507, 0.451], std=[0.269, 0.266, 0.256])
        ])
    
    image = transformEye(image)
    image = image.unsqueeze(0)
    return image

if __name__ == '__main__':
    #image_path = r'F:\gi4e_database\Leyes\053_07.png'
    root_path = r'C:\Users\Xuli\Desktop\MPIIGazeCode'
    json_path = r'C:\Users\Xuli\Desktop\MPIIGazeCode\all_data.json'
    with open(json_path, 'r') as f:
        meta = json.load(f)
    weight_path = r'C:\Users\Xuli\Desktop\hrnet_gi4e\Best\best.pth'
    model = NestedUNet(False)
    model.eval()
    model = load_model(model, weight_path)
    model = model.cuda()
    result = {"Train":[], "Val":[], "Test":[]}
    img_size = (64,64)
    for item in tqdm(meta['Train']):
        try:
            img_file = item[0]
            Leye_path = os.path.join(root_path, "LeftEye/{}".format(img_file))
            Reye_path = os.path.join(root_path, "RightEye/{}".format(img_file))
            Lkeypoints = predict_eye_center(model, Leye_path, weight_path, img_size=img_size)
            Rkeypoints = predict_eye_center(model, Reye_path, weight_path, img_size=img_size)
            item.append(Lkeypoints)
            item.append(Rkeypoints)
            result['Train'].append(item)
        except Exception as e:
            continue

    for item in tqdm(meta['Val']):
        try:
            img_file = item[0]
            Leye_path = os.path.join(root_path, "LeftEye/{}".format(img_file))
            Reye_path = os.path.join(root_path, "RightEye/{}".format(img_file))
            Lkeypoints = predict_eye_center(model, Leye_path, weight_path, img_size=img_size)
            Rkeypoints = predict_eye_center(model, Reye_path, weight_path, img_size=img_size)
            item.append(Lkeypoints)
            item.append(Rkeypoints)
            result['Val'].append(item)
        except Exception as e:
            continue
    for item in tqdm(meta['Test']):
        try:
            img_file = item[0]
            Leye_path = os.path.join(root_path, "LeftEye/{}".format(img_file))
            Reye_path = os.path.join(root_path, "RightEye/{}".format(img_file))
            Lkeypoints = predict_eye_center(model, Leye_path, weight_path, img_size=img_size)
            Rkeypoints = predict_eye_center(model, Reye_path, weight_path, img_size=img_size)
            item.append(Lkeypoints)
            item.append(Rkeypoints)
            result['Test'].append(item)
        except Exception as e:
            continue
    with open(r'C:\Users\Xuli\Desktop\MPIIGazeCode\all_data_with_eye.json','w') as nf:
        json.dump(result, nf)
             
        
        
        
