import math, shutil, os, time, argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model.hrnet import hrnet18_modified
from model.Unet import UNet

from loss import AdaptiveWingLoss

from dataset.gi4e import gi4e, gi4e_eye
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'F:\gi4e_database\blend')
parser.add_argument('--train_path', default=r'F:\gi4e_database\eye_train.json')
parser.add_argument('--val_path', default=r'F:\gi4e_database\eye_val.json')
parser.add_argument('--img_size', default=(64, 64))
parser.add_argument('--heatmap_size', default=(64, 64))
parser.add_argument('--pretrained_path', default=None)
parser.add_argument('--epochs', default=50)
parser.add_argument('--batch_size', default=4)
parser.add_argument('--num_workers', default=4)
parser.add_argument('--save_per_epoch', default=1)
args = parser.parse_args()

save_cp_and_plot = True
save_per_epoch = args.save_per_epoch

workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size

# train settings
lr = 0.001  
momentum = 0.9  
weight_decay = 1e-4  
lr_decay_milestones = [5, 10, 15, 20, 30]  
lr_decay_gamma = 0.1  


def main(model):
    model.cuda()
    cudnn.benchmark = True 
    dataTrain = gi4e_eye(data_path=args.data_path, json_path=args.train_path,sigma=1.0, img_size=args.img_size, heatmap_size=args.heatmap_size)
    dataVal = gi4e_eye(data_path=args.data_path, json_path=args.val_path,sigma=1.0, img_size=args.img_size, heatmap_size=args.heatmap_size)
   
   
    #criterion = nn.MSELoss().cuda()
    criterion = AdaptiveWingLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr,
                                weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                               milestones=lr_decay_milestones, 
                                               gamma=lr_decay_gamma)
    train(model, criterion, optimizer, scheduler, dataTrain, dataVal)


def train(model, criterion, optimizer, scheduler, dataTrain, dataVal):
    start_time = datetime.now().strftime("%m%d_%H%M")
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, prefetch_factor=2)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, prefetch_factor=2)
    print('Sucessfully load trainset:', len(train_loader))
    print('Sucessfully load valset:', len(val_loader))

    if args.pretrained_path:
        try:
            pretrained_dict = torch.load(args.pretrained_path)

            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print("Pretrained weights {} loaded successfully.".format(os.path.basename(args.pretrained_path)))
        except FileNotFoundError:
            print("Pretrained weights file not found! Training from scratch.")
        except Exception as e:
            print("Error loading pretrained weights:", e)
            print("Training from scratch.")

    global_step = 0
    e = []
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as bar:
            for i, (img, heatmap) in enumerate(train_loader):
                img = img.cuda()
                heatmap = heatmap.cuda()
               
                pred_hm = model(img)
                loss = criterion(pred_hm, heatmap)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # increment the total epoch loss
                epoch_loss += loss.item()

                # update the progress bar with the intermediate loss value
                bar.set_postfix(**{'loss per batch': loss.item()})
                bar.update()

                global_step += 1

            mean_epoch_loss = round(epoch_loss / len(train_loader), 5)
            print('\n', f"Epoch {epoch+1} Train loss:", mean_epoch_loss)
            train_loss.append(mean_epoch_loss)
            #train_mae.append(mean_epoch_mae)
            e.append(epoch+1)

        scheduler.step()
    
        mean_val_loss = validate(model, val_loader, criterion)
        print('\n', f"Epoch {epoch+1} Val Loss:", mean_val_loss)
        val_loss.append(mean_val_loss)

        if save_cp_and_plot:
            checkpoints_dir = f"results/{start_time}"
            os.makedirs(checkpoints_dir, exist_ok=True)
            #save_checkpoint(model, epoch, save_per_epoch, checkpoints_dir, best_val_loss, mean_val_loss)
            save_plot_jpg(e, train_loss,val_loss,checkpoints_dir)
            if (epoch+1) % save_per_epoch == 0:
                cp_name = 'Epoch' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, cp_name))
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best.pth'))
        #print(best_val_loss)
        
def validate(model, val_loader, criterion):
    if len(val_loader) == 0:
        print("NO VAL SET!")
        return 0, 0
    else:
        model.eval()
        val_loss=0
        with tqdm(total=len(val_loader), desc="Validation") as vbar:
            for i, (img, hm) in enumerate(val_loader):
                img = img.cuda()
                hm = hm.cuda()

                pred_hm = model(img)
                loss = criterion(pred_hm, hm)

                val_loss += loss.item()

                vbar.set_postfix(**{'loss per batch': loss.item()})
                vbar.update()

            mean_val_loss = round(val_loss / len(val_loader), 5)
        return mean_val_loss


def save_checkpoint(model, epoch, save_per_epoch, checkpoints_dir, best_val_loss, mean_val_loss):
    if (epoch+1) % save_per_epoch == 0:
        cp_name = 'Epoch' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, cp_name))
        if mean_val_loss < best_val_loss:
            best_val_loss = best_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'Best.pth'))

def save_plot_jpg(e, train_loss, val_loss, checkpoints_dir):
    plt.figure(figsize=(10,8))
    plt.plot(e, train_loss, linestyle="-", color="red", marker="o", label="train_loss")
    plt.plot(e, val_loss, linestyle="-", color="green", marker="o", label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoints_dir, "plot.jpg"), dpi=600)

if __name__ == "__main__":
    model = UNet()
    main(model)