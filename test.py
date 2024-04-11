import os
import time

import numpy as np
import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tqdm
from data_loader import ImgLoader
import timm
import albumentations as al
from albumentations.pytorch import ToTensorV2
import glob
import cv2
from models import get_model


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


device = torch.device("cuda:0")

model = get_model('resnet18', 5)

model.load_state_dict(torch.load('resuts/Apr11_10-55-02/weights/89_0.0625.pth'))
model.to(device)
model.eval()

resize_trans = al.Compose(
    [
        al.LongestMaxSize(max_size=224),
        al.PadIfNeeded(min_height=224, min_width=224, border_mode=0)])

norm_transform = al.Compose([
    al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])

resize_def_trans = al.Compose(
    [
        al.Resize(height=224, width=224)])

sigm = nn.Sigmoid()


def invert_bbox_transform(base_img, res_img, bbox):
    h, w, _ = base_img.shape
    h_r, w_r, _ = res_img.shape

    if h >= w:
        aspect_ratio = w / h
        w_after_long_max_size = w_r * aspect_ratio
        pad = w_r - w_after_long_max_size
        bbox[0] = bbox[0] - (pad / 2)
        bbox[2] = bbox[2] - (pad / 2)

        bbox[1] = bbox[1] * (h / h_r)
        bbox[3] = bbox[3] * (h / h_r)

        bbox[0] = bbox[0] * (w / w_after_long_max_size)
        bbox[2] = bbox[2] * (w / w_after_long_max_size)

    else:
        aspect_ratio = h / w
        h_after_long_max_size = h_r * aspect_ratio
        pad = h_r - h_after_long_max_size
        bbox[1] = bbox[1] - (pad / 2)
        bbox[3] = bbox[3] - (pad / 2)

        bbox[1] = bbox[1] * (h / h_after_long_max_size)
        bbox[3] = bbox[3] * (h / h_after_long_max_size)

        bbox[0] = bbox[0] * (w / w_r)
        bbox[2] = bbox[2] * (w / w_r)

    return bbox


def binary_acc(gt, pred):
    acc = accuracy_score(gt, pred)
    bac = balanced_accuracy_score(gt, pred)
    return acc, bac


test_pred = []
test_gt = []
ious = []
times = []

data_dir = '/home/anton/work/my/datasets/sob/ML/cats_dogs_dataset/valid/'
labels_pathes = glob.glob(data_dir + '*.txt')
for label_path in labels_pathes:

    bn = os.path.basename(label_path)
    file_name = bn[:-4]
    img_pth = os.path.join(data_dir, file_name + '.jpg')
    if os.path.exists(img_pth):
        file1 = open(label_path, "r")
        label_data = file1.read().split(' ')
        cls = label_data[0]
        xmin = int(label_data[1])
        ymin = int(label_data[2])
        xmax = int(label_data[3])
        ymax = int(label_data[4])

        orig_img = cv2.imread(img_pth)
        h, w, c = orig_img.shape

        t0 = time.time()
        img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        res_transform = resize_def_trans(image=img)
        res_image = res_transform["image"]

        norm_image = norm_transform(image=res_image)["image"]

        img = torch.unsqueeze(norm_image, dim=0)

        with torch.no_grad():
            out = model(img.to(device))

            predict_cls = (sigm(out[0, 0]) > 0.5).int().cpu().detach().numpy() + 1
            xmin_inscale = sigm(out[:, 1]).item() * 224
            ymin_inscale = sigm(out[:, 2]).item() * 224
            xmax_inscale = sigm(out[:, 3]).item() * 224
            ymax_inscale = sigm(out[:, 4]).item() * 224
            scaled_bbox = [xmin_inscale, ymin_inscale, xmax_inscale, ymax_inscale]

            inv_res = al.Compose(
                [
                    al.Resize(height=h, width=w)],
                bbox_params=al.BboxParams(format='pascal_voc'))
            scaled_bbox.append('0')
            s_transform = inv_res(image=res_image, bboxes=[scaled_bbox])
            res_image = s_transform["image"]
            bb = s_transform['bboxes'][0][0:4]

            t1 = time.time()
            times.append(t1- t0)

            # bb = invert_bbox_transform(orig_img, res_image, scaled_bbox)
            iou = bb_intersection_over_union(boxA=bb, boxB=[xmin, ymin, xmax, ymax])
            # cv2.rectangle(orig_img, (int(bb[0]), int(bb[1])),
            #               (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
            # cv2.imshow('asd', orig_img)
            # cv2.waitKey(0)

            ious.append(iou)
            test_pred.append(predict_cls)
            test_gt.append(int(cls))

print(binary_acc(test_gt, test_pred))
print(np.mean(ious))
print(np.mean(times))
print(len(ious))
