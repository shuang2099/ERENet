import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2 as cv
import time
from myloss import totalloss,bdcn_loss2
from net import mynet
import random
import kornia as kn
# 全局设备选择
device = torch.device('cuda:0')


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def compute_mean_bgr(dataset_path):
    mean_bgr = np.zeros(3)  # 初始化为零
    # 遍历数据集中的每个图像
    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        # 读取图像
        img = cv.imread(img_path)
        # 将图像转换为 float32 类型
        img = img.astype(np.float32)
        # 计算每个通道的均值
        mean_bgr += np.mean(img, axis=(0, 1,2))
    # 计算所有图像的平均值
    mean_bgr /= len(os.listdir(dataset_path))
    return mean_bgr


class TestDataset(Dataset):
    def __init__(self, root, test_root, tlabel_root, ):
        super(TestDataset, self).__init__()
        self.root = root
        self.test_root = test_root
        self.tlabel_root = tlabel_root
        self.img_height = 352
        self.img_width  = 352

        self.test_path = os.path.join(root, test_root)
        self.test_dir = os.listdir(self.test_path)
        self.tlabel_path = os.path.join(root, tlabel_root)
        self.tlabel_dir = os.listdir(self.tlabel_path)

        # Define other necessary variables like up_scale, mean_bgr, test_data
        self.up_scale = True  # Example value, please adjust accordingly
       # self.mean_bgr = compute_mean_bgr(os.path.join(root, test_root))
        self.mean_bgr = compute_mean_bgr(os.path.join(root, test_root))#[ 112.48125968, 113.34918299,94.5028383]
        # Use root as the path to compute mean
       # self.test_data = "CLASSIC"  # Example value, please adjust accordingly

    def __getitem__(self, idx):
        test_name = self.test_dir[idx]
        test_item_path = os.path.join(self.test_path, test_name)
        tlabel_name = self.tlabel_dir[idx]
        tlabel_item_path = os.path.join(self.tlabel_path, tlabel_name)

        X_test = cv.imread(test_item_path)
        y_test = cv.imread(tlabel_item_path, cv.IMREAD_GRAYSCALE)

        # Store original image size as integers
        orig_size = (X_test.shape[0], X_test.shape[1])  # Width, Height

        X_test, y_test = self.transform(X_test, y_test)

        return X_test, y_test, test_name, orig_size

    def transform(self, img, gt):

        img = np.array(img, dtype=np.float32)
        # bgr = np.mean(img,axis=(0,1))
        # # print(bgr)
        # img -= bgr

        if img.shape[0] < 650 or img.shape[1] < 650:
            img = cv.resize(img, (0, 0), fx=1.6, fy=1.6)
        elif img.shape[0] > 1024 or img.shape[1] > 1024:
            img = cv.resize(img, (0, 0), fx=0.8, fy=0.9)

        # Make sure images and labels are divisible by 2^4=16
        if img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv.resize(img, (img_width, img_height))

            gt = cv.resize(gt, (img_width, img_height))
        # else:
        #     pass
        #     img_width = self.img_width
        #     img_height = self.img_height
        #     img = cv.resize(img, (img_width, img_height))
        #     gt = cv.resize(gt, (img_width, img_height))

        # img = np.array(img, dtype=np.float32)
        # bgr = np.mean(img,axis=(0,1))
        # # img = np.array(img, dtype=np.float32)
        # # if self.rgb:
        # #     img = img[:, :, ::-1]  # RGB->BGR
        #
        bgr = np.mean(img,axis=(0,1))
        # print(bgr)
        img -= bgr
        # img -= bgr
        print(bgr)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        # gt = np.array(gt, dtype=np.float32)
        # gt /= 255.
        # gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


    def __len__(self):
        return len(self.test_dir)


# 定义数据集和数据加载器

model_path = 'model/model_biped.pth'
root_path = 'dataset/biped/'
test_path = 'test'
tlabel_path = 'labels'


test_dataset = TestDataset(root_path, test_path, tlabel_path, )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 创建模型实例并加载预训练权重

# 创建模型实例并加载预训练权重
model = mynet()
model = model.to(device)
state_dict = torch.load(model_path, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
model.load_state_dict(state_dict)
model.eval()

# 初始化测试损失
total_test_loss = 0.0


with torch.no_grad():
    start_time = time.time()  # Start time for FPS calculation
    for batch_idx, (X_test, y_test, test_name, orig_size) in enumerate(test_loader):
        # print(f"Processing image: {test_name}")
        X_test, y_test = X_test.to(device), y_test.to(device)
        # 前向传播获取预测结果
        y_pred = model(X_test)
        y_pred = y_pred[5]

        # X_test2 = X_test[:, [2, 1, 0], :, :]  # RGB
        # y_pred2 = model(X_test2)
        # print(y_pred.shape)
        # 计算损失
        # loss = totalloss(y_pred, y_test)
        # total_test_loss += loss.item()
        test_name = test_name[0]

        # 将预测图像保存为文件，使用 test_name 作为文件名
        y_pred_numpy = y_pred.cpu().numpy()  # 将张量移动到 CPU 并转换为 numpy 数组
        #gt[gt > 0.1] +=0.2#0.4
        y_pred_numpy [ y_pred_numpy > 0.8] = 1
        y_pred_numpy[y_pred_numpy < 0.2] = 0
        # y_pred_numpy[(0.3 < y_pred_numpy) & (y_pred_numpy < 0.4)] -= 0.3
        #
        y_pred_numpy = (y_pred_numpy[0, 0] * 255).astype('uint8')  # 转换图像数据范围
       # orig_size = tuple(orig_size)
        width = int(orig_size[0].item())  # Convert tensor to integer
        height = int(orig_size[1].item())  # Convert tensor to integer

        # Use the converted integers to construct the dsize tuple
        orig_size_int = (height, width)
        #
        y_pred_resized = cv.resize(y_pred_numpy, dsize=orig_size_int, interpolation=cv.INTER_LINEAR)
        #
        # 从test_name中移除文件扩展名（如果存在）
        test_name_without_extension = os.path.splitext(test_name)[0]
        # 构造pred_save_path时不添加任何扩展名
        pred_save_path = os.path.join(root_path, 'prebiped', test_name_without_extension)


        # 保存图像为PNG格式（考虑到之前的问题）
        cv.imwrite(f'{pred_save_path}.png', y_pred_resized)
    end_time = time.time()  # End time for FPS calculation
    elapsed_time = end_time - start_time
    fps = len(test_loader) / elapsed_time  # Calculate FPS
# 计算平均损失
average_test_loss = total_test_loss / len(test_loader)
print(f"Average Test Loss: {average_test_loss:.4f}")
print(f"Inference Speed (FPS): {fps:.2f}")
