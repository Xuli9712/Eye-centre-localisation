import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os, argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

from model.Unet import UNet
from model.Unetpp import NestedUNet

from loss import AdaptiveWingLoss

from dataset.gi4e import gi4e_eye
import matplotlib.pyplot as plt
import sys
from tools.heatmap_trans import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'D:\gi4e_database\blend')
parser.add_argument('--test_path', default=r'D:\gi4e_database\eye_test.json')
parser.add_argument('--model', default='UNet')
parser.add_argument('--loss', default='AdaWing')
parser.add_argument('--img_size', default=(32, 32))
parser.add_argument('--heatmap_size', default=(32, 32))
parser.add_argument('--sigma', default=1.0)
parser.add_argument('--weight_path', default=r'C:\Users\Xuli\Desktop\hrnet_gi4e\results\0719_0038\best.pth')
parser.add_argument('--batch_size', default=4)
parser.add_argument('--num_workers', default=4)
parser.add_argument('--deepsupervision', default=False)
args = parser.parse_args()

def main(model):
    model = load_model(model, args.weight_path)
    model = model.cuda()
    dataTest = gi4e_eye(data_path=args.data_path, json_path=args.test_path, sigma=1.0, img_size=args.img_size, heatmap_size=args.heatmap_size)

    if args.loss == 'MSE':
        criterion = nn.MSELoss().cuda()
    elif args.loss == 'AdaWing':
        criterion = AdaptiveWingLoss().cuda()

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)
    mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_frame = test(model, test_loader, criterion)
    return mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_frame

def test(model, test_loader, criterion):
    if len(test_loader) == 0:
        print("NO Test SET!")
        return 0, 0
    else:
        test_loss=0
        test_error=0
        inference_times = []
        with tqdm(total=len(test_loader), desc="Test") as vbar:
            for i, (img, heatmap, eye_pts, pupil_dist, Width, Height) in enumerate(test_loader):
                start_time = time.time() 
                img = img.cuda()
                heatmap = heatmap.cuda()
                eye_pts =eye_pts.cuda()
                pupil_dist = pupil_dist.cuda()
                Width = Width.cuda()
                Height = Height.cuda()
                pupil_dist = pupil_dist.view(-1, 1)

                pred_hm = model(img)
                if args.deepsupervision:
                    loss = 0
                    for pred_one_hm in pred_hm:
                        one_loss = criterion(pred_one_hm, heatmap)
                        loss += one_loss
                    pred_hm = pred_hm[3]   
                else:
                    loss = criterion(pred_hm, heatmap)
                
                error, norm_error = cal_batch_error(pred_hm, eye_pts, Width, Height, pupil_dist)
                

                test_loss += loss.item()
                test_error += norm_error.cpu().item()
                end_time = time.time() 
                inference_times.append(end_time - start_time) 

                vbar.set_postfix(**{'loss per batch': loss.item()})
                vbar.update()

            mean_test_loss = round(test_loss / len(test_loader), 5)
            mean_test_error = round(test_error / len(test_loader), 5)
            mean_inference_time = round(sum(inference_times) / len(inference_times), 5)
            mean_inference_time_per_frame = round(mean_inference_time/int(args.batch_size), 5)
        print("Test Loss: ", mean_test_loss, "Test Error", mean_test_error, "Inference Time(s) Per Batch({}):".format(args.batch_size), mean_inference_time, "Mean Inference Time(s) Per Frame:", mean_inference_time_per_frame)

        return mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_frame


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



if __name__ == "__main__":
    if args.model == 'UNet':
        model = UNet()
    elif args.model == 'UNet++':
        model =NestedUNet(args.deepsupervision)
    model.eval()
    mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_frame = main(model)



