import os
import cv2
import albumentations as al
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import glob


class ImgLoader(Dataset):
    def __init__(self, data_dir, train=True):
        self.data = []
        self.mapping = {'1': 0, '2': 1}
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

                self.data.append([img_pth, self.mapping[cls], [xmin, ymin, xmax, ymax]])
            else:
                print(img_pth)

        self.train = train
        self.h = 224
        self.w = 224

        self.transform = al.Compose([al.HorizontalFlip(p=0.5),
                                     al.BBoxSafeRandomCrop(p=0.5),
                                     al.HueSaturationValue(p=0.5),
                                     al.RandomBrightnessContrast(p=0.5)
                                     ], bbox_params=al.BboxParams(format='pascal_voc', ))
        self.val_transform = al.Compose([
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()])

        # self.resize = al.Compose(
        #     [
        #         al.LongestMaxSize(max_size=max(self.h, self.w)),
        #         al.PadIfNeeded(min_height=self.h, min_width=self.w, border_mode=0)],
        #     bbox_params=al.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.resize = al.Compose(
            [
                al.Resize(height=self.h, width=self.w)],
            bbox_params=al.BboxParams(format='pascal_voc'))

        self.classes = []
        for val in self.data:
            self.classes.append(val[1])

    def __len__(self):
        return len(self.data)

    def make_class_weights(self):
        pos = sum(self.classes)
        all = len(self.classes)
        neg = all - pos

        weights = neg / pos
        return weights

    def __getitem__(self, item):

        img_path = self.data[item][0]
        label = int(self.data[item][1])
        bbox = self.data[item][2]
        bbox.append('0')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.train:

            f_transform = self.transform(image=img, bboxes=[bbox])
            image = f_transform["image"]
            trf_box = f_transform['bboxes']

            s_transform = self.resize(image=image, bboxes=trf_box)
            res_image = s_transform["image"]
            res_trf_box = s_transform['bboxes']

            sclased_xmin = res_trf_box[0][0] / self.w
            sclased_ymin = res_trf_box[0][1] / self.h
            sclased_xmax = res_trf_box[0][2] / self.w
            sclased_ymax = res_trf_box[0][3] / self.h

            # cv2.rectangle(res_image, (int(res_trf_box[0][0]), int(res_trf_box[0][1])),
            #               (int(res_trf_box[0][2]), int(res_trf_box[0][3])), (255, 0, 0), 2)
            # cv2.imshow('asd', res_image)
            # cv2.waitKey(0)

            image = self.val_transform(image=res_image)["image"]
            return image, label, sclased_xmin, sclased_ymin, sclased_xmax, sclased_ymax
        else:
            res_transform = self.resize(image=img, bboxes=[bbox])
            res_image = res_transform["image"]
            res_trf_box = res_transform['bboxes']

            sclased_xmin = res_trf_box[0][0] / self.w
            sclased_ymin = res_trf_box[0][1] / self.h
            sclased_xmax = res_trf_box[0][2] / self.w
            sclased_ymax = res_trf_box[0][3] / self.h

            image = self.val_transform(image=res_image)["image"]
            return image, label, sclased_xmin, sclased_ymin, sclased_xmax, sclased_ymax


if __name__ == '__main__':
    path = '/home/anton/work/my/datasets/sob/ML/cats_dogs_dataset/valid/'
    a = ImgLoader(data_dir=path)
    for i in range(0, 100):
        m = a[i]
