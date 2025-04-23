import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json


class TestDataset(Dataset):
    test_modes = [ 'test', ]
    dataset_types = ['rgbr', ]

    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 test_mode='test',
                 dataset_type='rgbr',
                 ):

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.test_mode = test_mode
        self.dataset_type = dataset_type

        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        #rint(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        assert self.test_mode in self.test_modes, self.test_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []

        file_path = os.path.join(data_root, self.test_list)
        with open(file_path) as f:
            files = json.load(f)
        for pair in files:
            if isinstance(pair[0], list):  # 检查是否为嵌套列表
                for sub_pair in pair:  # 遍历嵌套列表
                    tmp_img = sub_pair[0]  # 假设每个子列表的第一个元素是图像路径
                    tmp_gt = sub_pair[1]  # 假设每个子列表的第二个元素是标签路径
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (os.path.join(self.data_root, tmp_img),
                     os.path.join(self.data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        img = np.array(img, dtype=np.float32)
        bgr = np.mean(img, axis=(0, 1))
        img -= bgr
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width,img_height))
            gt = None

        # # Make images and labels at least 512 by 512
        # elif img.shape[0] < 512 or img.shape[1] < 512:
        #     img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height)) # 512
        #     gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height)) # 512

        # Make sure images and labels are divisible by 2^4=16
        else:
            i_h, i_w, _ = img.shape
            crop_size = self.img_height if self.img_height == self.img_width else None
            if i_w > crop_size and i_h > crop_size:
                i = random.randint(0, i_h - crop_size)
                j = random.randint(0, i_w - crop_size)
                img = img[i:i + crop_size, j:j + crop_size]
                gt = gt[i:i + crop_size, j:j + crop_size]
            else:
                # New addidings
                img = cv2.resize(img, dsize=(crop_size, crop_size))
                gt = cv2.resize(gt, dsize=(crop_size, crop_size))
       # if self.yita is not None:
        #     gt[gt >= self.yita] = 1

        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        # img=cv2.resize(img, (400, 464))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            # gt[gt > 0] += 0.2 # 0.4
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]
    def __init__(self,
                 data_root,
                 train_data,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_list,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 ):
        self.data_root = data_root
        self.train_data = train_data
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.train_list = train_list
        self.crop_img = crop_img

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []

        file_path = os.path.join(data_root, self.train_list)

        with open(file_path) as f:
            files = json.load(f)
        for pair in files:
            if isinstance(pair[0], list):  # 检查是否为嵌套列表
                for sub_pair in pair:  # 遍历嵌套列表
                    tmp_img = sub_pair[0]  # 假设每个子列表的第一个元素是图像路径
                    tmp_gt = sub_pair[1]  # 假设每个子列表的第二个元素是标签路径
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (os.path.join(self.data_root, tmp_img),
                     os.path.join(self.data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        bgr = np.mean(img, axis=(0,1))
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        #img -= self.mean_bgr
        img -= bgr
        i_h, i_w,_ = img.shape
        # data = []
        # f self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None#448# MDBD=480 BIPED=480/400 BSDS=352

        # for BSDS 352/BRIND      # img = cv2.resize(img, dsize=(crop_size,crop_size))
        # gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # if i_w > crop_size and i_h>crop_size:
        #     i = random.randint(0, i_h - crop_size)
        #     j = random.randint(0, i_w - crop_size)
        #     img = img[i:i + crop_size , j:j + crop_size ]
        #     gt = gt[i:i + crop_size , j:j + crop_size ]
        #     img = cv2.resize(img, dsize=(352, 352))
        #     gt = cv2.resize(gt, dsize=(352, 352))
        # else:
        #       img = cv2.resize(img, dsize=(352, 352))
        #       gt = cv2.resize(gt, dsize=(352, 352))
        # # for BIPED/MDBD
        # img = cv2.resize(img, dsize=(crop_size,crop_size))
        # gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        if i_w > 250 and i_h > 250:  # before 420
            h, w = gt.shape
            if np.random.random() > 0.2:  # before i_w> 500 and i_h>500:
                LR_img_size = crop_size  # l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size, j:j + LR_img_size]
                gt = gt[i:i + LR_img_size, j:j + LR_img_size]
            else:
                LR_img_size = 196# 256 300 400  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
                i = random.randint(0, h - LR_img_size)
                j = random.randint(0, w - LR_img_size)
                # if img.
                img = img[i:i + LR_img_size, j:j + LR_img_size]
                gt = gt[i:i + LR_img_size, j:j + LR_img_size]
                img = cv2.resize(img, dsize=(crop_size, crop_size), )
                gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:  # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # BRIND
        # gt[gt > 0.] = 1#0.4
        gt[gt > 0.] += 0.2  # 0.5 for BIPED
        gt = np.clip(gt, 0., 1.)
        # gt[gt > 0.1] =1#0.4
        # gt = np.clip(gt, 0., 1.)        # # for BIPED
        # gt[gt > 0.2] += 0.6# 0.5 for BIPED
        # gt = np.clip(gt, 0., 1.) # BIPED
        # # for MDBD
        # gt[gt > 0.1] +=0.7
        # gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt
