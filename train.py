import os

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
from models import get_model

'''
STEP 0: initial
'''

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

experiment_name = current_time

out_dir = 'resuts'

log_path = os.path.join(out_dir, experiment_name, 'log')
save_weight_path = os.path.join(out_dir, experiment_name, 'weights')

os.makedirs(os.path.join(out_dir, experiment_name), exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_weight_path, exist_ok=True)

device = torch.device("cuda:0")
writer = SummaryWriter(logdir=log_path)

train_ds = ImgLoader('/home/anton/work/my/datasets/sob/ML/cats_dogs_dataset/train/')
test_ds = ImgLoader('/home/anton/work/my/datasets/sob/ML/cats_dogs_dataset/valid/', train=False)

cls_weights = torch.tensor(train_ds.make_class_weights(), dtype=torch.float)
print(f'class weights -- {cls_weights}')
epoch = 250
batch_size = 256
lr = 0.001


def binary_acc(gt, pred):
    acc = accuracy_score(gt, pred)
    bac = balanced_accuracy_score(gt, pred)
    return acc, bac


model = get_model('resnet18', 5)

model.to(device)
criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_weights]).to(device))
criterion_regr = nn.L1Loss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, pin_memory=False, shuffle=True,
    num_workers=0, drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, pin_memory=False, shuffle=True,
    num_workers=0, drop_last=False
)

print(len(train_ds), len(test_ds))
sigm = nn.Sigmoid()

maximum = 100
for epoch in range(epoch):

    model.train()

    train_losses_regr = []
    train_losses_cls = []
    train_losses = []

    train_pred = []
    train_gt = []

    for imgs, gt_cls, xmin_gt, ymin_gt, xmax_gt, ymax_gt in tqdm.tqdm(train_loader):
        gt_cls = gt_cls.float().to(device)
        xmin_gt = xmin_gt.float().to(device)
        ymin_gt = ymin_gt.float().to(device)
        xmax_gt = xmax_gt.float().to(device)
        ymax_gt = ymax_gt.float().to(device)

        optimizer.zero_grad()

        out = model(imgs.to(device)).squeeze(1)

        loss_cls = criterion_cls(out[:, 0], gt_cls)
        loss_regr = (criterion_regr(sigm(out[:, 1]), xmin_gt) +
                     criterion_regr(sigm(out[:, 2]), ymin_gt) +
                     criterion_regr(sigm(out[:, 3]), xmax_gt) +
                     criterion_regr(sigm(out[:, 4]), ymax_gt))

        loss = loss_cls + loss_regr
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_losses_regr.append(loss_regr.item())
        train_losses_cls.append(loss_cls.item())

        predicts = (sigm(out[:, 0]) > 0.5).int().cpu().detach().numpy()
        train_pred.extend(list(np.squeeze(predicts).astype(int)))
        train_gt.extend(list(np.squeeze(np.round(gt_cls.cpu().detach().numpy().astype(int)))))

    scheduler.step()

    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)

    test_losses_regr = []
    test_losses_cls = []
    test_losses = []

    test_pred = []
    test_gt = []

    model.eval()
    with torch.no_grad():
        for imgs, gt_cls, xmin_gt, ymin_gt, xmax_gt, ymax_gt in tqdm.tqdm(test_loader):
            gt_cls = gt_cls.float().to(device)
            xmin_gt = xmin_gt.float().to(device)
            ymin_gt = ymin_gt.float().to(device)
            xmax_gt = xmax_gt.float().to(device)
            ymax_gt = ymax_gt.float().to(device)

            out = model(imgs.to(device)).squeeze(1)

            loss_cls = criterion_cls(out[:, 0], gt_cls)
            loss_regr = (criterion_regr(sigm(out[:, 1]), xmin_gt) +
                         criterion_regr(sigm(out[:, 2]), ymin_gt) +
                         criterion_regr(sigm(out[:, 3]), xmax_gt) +
                         criterion_regr(sigm(out[:, 4]), ymax_gt))

            loss = loss_cls + loss_regr

            test_losses.append(loss.item())
            test_losses_regr.append(loss_regr.item())
            test_losses_cls.append(loss_cls.item())

            predicts = (sigm(out[:, 0]) > 0.5).int().cpu().detach().numpy()
            test_pred.extend(list(np.squeeze(predicts).astype(int)))
            test_gt.extend(list(np.squeeze(np.round(gt_cls.cpu().detach().numpy().astype(int)))))

    mean_train_losses = np.mean(train_losses).round(5)
    mean_test_losses = np.mean(test_losses).round(5)

    mean_train_losses_regr = np.mean(train_losses_regr).round(5)
    mean_test_losses_regr = np.mean(test_losses_regr).round(5)

    mean_train_losses_cls = np.mean(train_losses_cls).round(5)
    mean_test_losses_cls = np.mean(test_losses_cls).round(5)

    mean_train_acc, mean_train_bac = binary_acc(train_gt, train_pred)
    mean_validation_acc, mean_validation_bac = binary_acc(test_gt, test_pred)

    print(f'{epoch}, '
          f'mean_train_losses -- {np.round(mean_train_losses, 4)}, '
          f'mean_test_losses -- {np.round(mean_test_losses, 4)}, '
          f'mean_train_acc -- {np.round(mean_train_acc, 4)}, '
          f'mean_train_bac -- {np.round(mean_train_bac, 4)},'
          f'mean_validation_acc -- {np.round(mean_validation_acc, 4)}',
          f'mean_validation_bac -- {np.round(mean_validation_bac, 4)}',
          f'mean_train_losses_regr -- {np.round(mean_train_losses_regr, 4)}, '
          f'mean_test_losses_regr -- {np.round(mean_test_losses_regr, 4)} ',
          f'mean_train_losses_cls -- {np.round(mean_train_losses_cls, 4)}',
          f'mean_test_losses_cls -- {np.round(mean_test_losses_cls, 4)}',

          )

    writer.add_scalar('mean_train_losses', mean_train_losses, epoch)
    writer.add_scalar('mean_test_losses', mean_test_losses, epoch)
    writer.add_scalar('mean_train_acc', mean_train_acc, epoch)
    writer.add_scalar('mean_train_bac', mean_train_bac, epoch)
    writer.add_scalar('mean_validation_acc', mean_validation_acc, epoch)
    writer.add_scalar('mean_validation_bac', mean_validation_bac, epoch)
    writer.add_scalar('mean_train_losses_regr', mean_train_losses_regr, epoch)
    writer.add_scalar('mean_test_losses_regr', mean_test_losses_regr, epoch)
    writer.add_scalar('mean_train_losses_cls', mean_train_losses_cls, epoch)
    writer.add_scalar('mean_test_losses_cls', mean_test_losses_cls, epoch)

    if (1 - mean_validation_bac) + mean_test_losses_regr < maximum:
        maximum = (1 - mean_validation_bac) + mean_test_losses_regr
        torch.save(model.state_dict(),
                   os.path.join(save_weight_path,
                                f'{epoch}_{np.round(maximum, 4)}.pth'))
