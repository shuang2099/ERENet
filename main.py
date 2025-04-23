import torch
torch.autograd.set_detect_anomaly(True)
import torch.optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import *
from myloss import totalloss, bdcn_loss2
from net import mynet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
torch.backends.cudnn.enabled = True# 定义参数

epochs = 200
lr = 0.0005
BATCH_SIZE = 12
#momentum = 0.9
weight_decay = 0.00002
step_size = 2

def train(model, train_loader, test_loader, epochs, optimizer, calculate_loss,  device, step_size):
    train_loss_all = []
    test_loss_all = []
    no_improvement_count = 0
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0

        # 初始化种子值
        seed = 1009

        # 每5个epoch使用特定的种子来打乱数据，并在下一次打乱时种子数加一
        if epoch % 5 == 0 and epoch > 0:  # 确保不在第一个epoch打乱，除非这是你的意图
            np.random.seed(seed)
            torch.manual_seed(seed)
            indices = np.random.permutation(len(train_dataset))
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            # 更新种子值，以便下次使用时种子数加一
            seed += 1

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, ncols=100) as progress_bar:
            for batch in progress_bar:  # batch是从DataLoader返回的字典
                X = batch['images'].to(device)  # 现在正确地处理张量数据
                y = batch['labels'].to(device)

                y_pre = model(X)

                loss = calculate_loss(y_pre, y, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f}")
                progress_bar.update(1)

            train_loss_normalized = train_loss / (len(train_loader))
            train_loss_all.append(train_loss_normalized)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:  # batch是从DataLoader返回的字典
                    X_test = batch['images'].to(device)  # 现在正确地处理张量数据
                    y_test = batch['labels'].to(device)
                    y_pred = model(X_test)

                    loss = calculate_loss(y_pred, y_test, device)

                    test_loss += loss.item()
            average_test_loss = test_loss / (len(test_loader))
            test_loss_all.append(average_test_loss)

            tqdm.write(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_normalized:.4f} | Test Loss: {average_test_loss:.4f}")


            # 在每个epoch结束后保存模型的状态字典
            save_path =  f'/media/h3c/0ef529db-0550-4e97-9dda-4d84a6552837/chenyicheng/model/modelbnb1_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)

            # 新增：判断测试集损失是否连续若干个epoch未下降，如果是，则降低学习率
            if epoch > 0 and average_test_loss > test_loss_all[-2]:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            min_lr = 0.00002  # 设置最小学习率
            if no_improvement_count >= step_size:
                current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
                if current_lr > min_lr:
                    new_lr = max(current_lr * 0.2, min_lr)  # 计算新学习率但不低于min_lr
                    for param_group in optimizer.param_groups:
                     param_group['lr'] = new_lr  # 更新学习率
                    print(f"Reduced learning rate to {new_lr}")
                    no_improvement_count = 0  # 重置无改善计数
                else:
                    print("Learning rate already at or below minimum threshold. Not reducing further.")

    # save_path = f'../cyc/model/data/model1.pth'
    # torch.save(model, save_path)
    #print(f"训练好的模型状态字典已保存在 {save_path}")

    return train_loss_all, test_loss_all
if __name__ == "__main__":
    #data_root = '../cyc/data//BIPED22/BIPED/'
    #data_root2 = '../cyc/data/BRIND-main/BRIND-main/'
    data_root =  f'/media/h3c/0ef529db-0550-4e97-9dda-4d84a6552837/BIPED0'
    #data_root2 =  f'/media/h3c/0ef529db-0550-4e97-9dda-4d84a6552837/BRIND-main/BRIND-main/'

    train_dataset = BipedDataset(data_root=data_root,
                                 train_data='biped',
                               #rain_data='bsds500',
                                 #mean_bgr=[103.939,116.779,123.68],
                                 #mean_bgr=[104.007,116.669,122.679],
                                 mean_bgr=[159.81,159.87, 162.72],
                                 img_width=352,
                                 img_height=352,
                                 train_mode='train',
                                 train_list='train_pair.lst')
    test_dataset = TestDataset(data_root=data_root,
                               test_data='biped',
                               #test_data='bsds500',
                               #mean_bgr=[103.939, 116.779, 123.68],#biped
                               #mean_bgr=[104.007,116.669,122.679],#bsds500
                               mean_bgr=[159.81,159.87, 162.72],
                               img_width=640,
                               img_height=640,
                               test_list='test_pair.lst' )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = bwdn(n, channels).to(device)
    model = bwdn().to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr, weight_decay=weight_decay)


    def calculate_loss(y_pre, y, device):
        loss = torch.zeros(1).to(device)
        loss = loss + bdcn_loss2(y_pre[0], y, l_weight=1.1)
        loss = loss + bdcn_loss2(y_pre[1], y, l_weight=1.1)
        loss = loss + totalloss(y_pre[2], y, tex_factor=0.005, bdr_factor=2, balanced_w=1.1)
        loss = loss + totalloss(y_pre[3], y, tex_factor=0.005, bdr_factor=2, balanced_w=1.1)
        # loss5 = totalloss(y_pre[4], y, tex_factor=0.005, bdr_factor=2, balanced_w=1.1)
        loss = loss + totalloss(y_pre[4], y, tex_factor=0.005, bdr_factor=4, balanced_w=1.3)
        loss = loss + totalloss(y_pre[5], y, tex_factor=0.005, bdr_factor=4, balanced_w=1.5)
        return loss

    # 训练
    train_loss_all, test_loss_all = train(model, train_loader, test_loader, epochs, optimizer, calculate_loss, device,step_size=step_size)


    # 绘制训练和测试损失曲线
    plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_loss_all, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
