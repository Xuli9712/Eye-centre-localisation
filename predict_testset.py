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


def predict_eye_center_testset(model, img_path, weight_path,keypoints_gt, img_size):
    # Load and preprocess the image
    image = Image.open(img_path)
    img_copy1 = image.copy()
    img_copy2 = image.copy()
    W, H = image.size
    W = torch.tensor(W).cuda()
    H = torch.tensor(H).cuda()
    
    image = preprocess_image(image, img_size)

    image = image.cuda()

    with torch.no_grad():
        output = model(image)

    
    num_keypoints = output.shape[0]
    heatmap_size = output.shape[1:]

    keypoints = heatmap2coord(output, W, H).squeeze().cpu().numpy()

    #print(keypoints)

    for i in range(3):
        img_copy2 = cv2.circle(np.array(img_copy2), tuple(map(int, keypoints_gt[i])), 1, (0, 0, 255), -1)
        img_copy2 = cv2.circle(np.array(img_copy2), tuple(map(int, keypoints[i])), 1, (255, 0, 0), -1)
        
    
    output = output.squeeze().cpu().numpy()
    
     # show the image with keypoints 
    plt.subplot(1, 5 , 1)
    plt.imshow(img_copy1)
    plt.title(f"Input")

    
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
    label_path = r'D:\gi4e_database\eye_test.json'
    with open(label_path, 'r') as ts:
        test_label = json.load(ts)
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
    for item in tqdm(test_label):
        img_name = item[0]
        keypoints_gt = item[1]
        image_path = os.path.join(root_path, img_name)
        result_image, keypoints = predict_eye_center_testset(model, image_path, weight_path, keypoints_gt, img_size=(64, 64))
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image_path = os.path.join(result_path, img_name)
        cv2.imwrite(result_image_path, result_image_bgr)


        
