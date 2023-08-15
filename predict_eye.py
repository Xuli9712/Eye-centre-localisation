import cv2
import torch
import numpy as np
import sys, json
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import functional as F
from model.Unet import UNet
from model.Unetpp import NestedUNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tools.heatmap_trans import *
from tqdm import tqdm 

def predict_eye_center(model, img_path, weight_path, img_size):
    # Load and preprocess the image
    image = Image.open(img_path)
    img_copy1 = image.copy()
    img_copy2 = image.copy()
    W, H = image.size
    #img_copy = img_copy.resize(img_size)
    W = torch.tensor(W).cuda()
    H = torch.tensor(H).cuda()
    
    image = preprocess_image(image, img_size)

    image = image.cuda()

    with torch.no_grad():
        output = model(image)

    #output = output.squeeze().cpu().numpy()
    
    num_keypoints = output.shape[0]
    heatmap_size = output.shape[1:]

    #keypoints = extract_keypoints_from_heatmap(output)
    #keypoints = extract_keypoints_from_heatmap_to_ori_size(output, W, H)
    keypoints = heatmap2coord(output, W, H).squeeze().cpu().numpy()
    #print(keypoints)

    for i in range(3):
        img_copy2 = cv2.circle(np.array(img_copy2), tuple(map(int, keypoints[i])), 1, (0, 0, 255), -1)
    
    output = output.squeeze().cpu().numpy()
    
     # 显示左侧的单张图像, 点已经绘制上去了
    plt.subplot(1, 5 , 1)
    plt.imshow(img_copy1)
    plt.title(f"Input")
    '''
    # 显示右侧的热力图
    for i in range(3):
        plt.subplot(1, 5, i+3)
        plt.imshow(output[i], cmap='hot')
        plt.title(f"Target {i+1}")
    '''
    
    for i in range(3):
        plt.subplot(1, 5, i+2)
        plt.imshow(output[i], cmap='hot')
        plt.title(f"Output{i+1}")
    
    plt.subplot(1, 5 , 5)
    plt.imshow(img_copy2)
    plt.title(f"Result")
    
    plt.tight_layout()  
    plt.show()

    return img_copy2, keypoints




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

def predict_eye_center_module(processed_image, image_W, image_H):
    weight_path = r'C:\Users\Xuli\Desktop\hrnet_gi4e\results\0719_0038\best.pth'
    model = UNet()
    model.eval()
    model = load_model(model, weight_path)

    model = model.cuda()
    image = image.cuda()

    with torch.no_grad():
        outputs = model(processed_image)

    output = outputs[3].squeeze().cpu().numpy()
    
    num_keypoints = output.shape[0]
    heatmap_size = output.shape[1:]

    #keypoints = extract_keypoints_from_heatmap(output)
    keypoints = extract_keypoints_from_heatmap_to_ori_size(output, image_W, image_H)
    print(keypoints)

    for i in range(num_keypoints):
        img_copy = cv2.circle(np.array(img_copy), tuple(map(int, keypoints[i])), 1, (0, 0, 255), -1)

    # 显示左侧的单张图像, 点已经绘制上去了
    plt.subplot(1, 13 , 1)
    plt.imshow(img_copy)
    '''
    # 显示右侧的热力图
    for i in range(3):
        plt.subplot(1, 5, i+3)
        plt.imshow(output[i], cmap='hot')
        plt.title(f"Target {i+1}")
    '''
    
    for j, output in enumerate(outputs):
        for i in range(3):
            plt.subplot(1, 5, i+2+3*j)
            plt.imshow(output[j][i], cmap='hot')
            plt.title(f"Target{j+1} {i+1}")

    plt.tight_layout()  
    plt.show()
    return img_copy, keypoints

def preprocess_image(image, img_size):
    transformEye = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.472, 0.333, 0.342], std=[0.209, 0.163, 0.166])
        ])
    
    image = transformEye(image)
    image = image.unsqueeze(0)
    return image


if __name__ == '__main__':
    #image_path = r'F:\gi4e_database\Leyes\053_07.png'
    root_path = r'D:\gi4e_database\blend'
    img_list = os.listdir(root_path)
    weight_path = r'Best\best.pth'
    result_path = os.path.join(root_path, 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #model = UNet()
    model = NestedUNet(False)
    # eval mode
    model.eval()
    model = load_model(model, weight_path)
    model = model.cuda()
    for img_name in tqdm(img_list):
        image_path = os.path.join(root_path, img_name)
        result_image, keypoints = predict_eye_center(model, image_path, weight_path, img_size=(64, 64))
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image_path = os.path.join(result_path, img_name)
        cv2.imwrite(result_image_path, result_image_bgr)

        
