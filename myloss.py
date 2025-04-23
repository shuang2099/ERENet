
import torch.nn.functional as F
import torch
import torch.nn as nn

device = torch.device("cuda:0")
def bdrloss(y_pre, y, radius):
    filt = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1)
    filt.requires_grad = False
    filt = filt.to(device)
    bdr_pred = y_pre * y
    pred_bdr_sum = y * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(y.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[y == 1] = 0
    pred_texture_sum = F.conv2d(y_pre * (1 - y) * mask, filt, bias=None, stride=1, padding=radius)
    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -y * torch.log(softmax_map)
    cost[y == 0] = 0
    return torch.sum(cost.float().mean((1, 2, 3)))

def textureloss(y_pre, y, mask_radius):
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1= filt1.to(device)
    
    filt2 = torch.ones(1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(y_pre.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(y.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))

def totalloss(y_pre, y, tex_factor=0.002, bdr_factor=4, balanced_w=1.1):
    y = y.float()
    # y = torch.sigmoid(y)
    y_pre = y_pre.float()
    # y_pre = (y_pre - y_pre.min()) / ( y_pre.max() -  y_pre.min())
    #y_pre = torch.sigmoid(y_pre)
    with torch.no_grad():
        mask = y.clone()

        num_positive = torch.sum((mask == 1.0).float()).float()
        num_negative = torch.sum((mask == 0.0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
        y_pre.float(), y.float(), weight=mask,  reduction='none')
    cost = torch.sum(cost.float().mean((1, 2, 3)))
    # cost2 = torch.sum(torch.nn.functional.mse_loss(
    #             y_pre.float(), y.float(), reduction='none'))
    label_w = (y != 0).float()
    textcost = textureloss(y_pre.float(), label_w.float(), mask_radius=4)
    bdrcost = bdrloss(y_pre.float(), label_w.float(), radius=4)
    return cost + bdr_factor * bdrcost + tex_factor * textcost
    # return bdrcost
def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    #inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost
# def pLoss(y_pre, y):
#         # 计算非零元素的个数
#         non_zero_count = torch.sum(y != 0).float()
#         zero_count = torch.sum(y == 0).float()
#         tot = non_zero_count + zero_count
#         # 计算正负样本的损失
#         positive_loss = torch.sum((y != 0).float() * (y_pre - y) ** 2)
#         negative_loss = torch.sum((y == 0).float() * (y_pre - y) ** 2)
#         out = zero_count / tot * negative_loss + non_zero_count / tot * positive_loss
#         return out
# def totalloss2(y_pre, y, tex_factor=0.3, bdr_factor=1.2, balanced_w=1):
#     y = y.float()
#     # y = torch.sigmoid(y)
#     y_pre = y_pre.float()
#     # y_pre = torch.sigmoid(y_pre)
#     with torch.no_grad():
#         mask = y.clone()
#
#         num_positive = torch.sum((mask == 1).float()).float()
#         num_negative = torch.sum((mask == 0).float()).float()
#         beta = num_negative / (num_positive + num_negative)
#         mask[mask == 1] = beta
#         mask[mask == 0] = balanced_w * (1 - beta)
#         mask[mask == 2] = 0
#
#
#     # cost = torch.sum(torch.nn.functional.binary_cross_entropy(
#     #     y_pre.float(), y.float(), weight=mask,  reduction='none'))
#     cost = torch.sum(torch.nn.functional.mse_loss(
#                 y_pre.float(), y.float(), reduction='none'))
#     label_w = (y != 0).float()
#
#     textcost = textureloss(y_pre.float(), label_w.float(), mask_radius=2)
#     bdrcost = bdrloss(y_pre.float(), label_w.float(), radius=2)
#     pcost = pLoss(y_pre.float(),y.float())
#
#     return  cost+bdr_factor * bdrcost+pLoss+pcost

# def hed_loss2(inputs, targets, l_weight=1.1):
#     # bdcn loss with the rcf approach
#     targets = targets.long()
#     mask = targets.float()
#     num_positive = torch.sum((mask > 0.1).float()).float()
#     num_negative = torch.sum((mask <= 0.).float()).float()
#
#     mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
#     inputs= torch.sigmoid(inputs)
#     cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), targets.float())
#
#     return l_weight*torch.sum(cost)
#
#
# def bdcn_loss2(inputs, targets, l_weight=1.1):
#     # bdcn loss with the rcf approach
#     targets = targets.long()
#     # mask = (targets > 0.1).float()
#     mask = targets.float()
#     num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
#     num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1
#
#     mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
#     mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
#     # mask[mask == 2] = 0
#     inputs= torch.sigmoid(inputs)
#     cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
#     # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
#     cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
#     return l_weight*cost
#
# def bdcn_lossORI(inputs, targets, l_weigts=1.1,):
#     """
#     :param inputs: inputs is a 4 dimensional data nx1xhxw
#     :param targets: targets is a 3 dimensional data nx1xhxw
#     :return:
#     """
#     n, c, h, w = inputs.size()
#     # print(cuda)
#     weights = np.zeros((n, c, h, w))
#     for i in range(n):
#         t = targets[i, :, :, :].data.numpy()
#         t = t.to(device)
#         pos = (t == 1).sum()
#         neg = (t == 0).sum()
#         valid = neg + pos
#         weights[i, t == 1] = neg * 1. / valid
#         weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
#     weights = torch.Tensor(weights)
#     # if cuda:
#     weights = weights.cuda()
#     inputs = torch.sigmoid(inputs)
#     loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())
#     return l_weigts*loss
#
# def rcf_loss(inputs, label):
#
#     label = label.long()
#     mask = label.float()
#     num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
#     num_negative = torch.sum((mask == 0).float()).float()
#
#     mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
#     mask[mask == 2] = 0.
#     inputs= torch.sigmoid(inputs)
#     cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())
#
#     return 1.*torch.sum(cost)
#
# # ------------ cats losses ----------
#
# def bdrloss(prediction, label, radius,):
#     '''
#     The boundary tracing loss that handles the confusing pixels.
#     '''
#
#     filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
#     filt.requires_grad = False
#     filt = filt.to(device)
#
#     bdr_pred = prediction * label
#     pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
#     texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
#     mask = (texture_mask != 0).float()
#     mask[label == 1] = 0
#     pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)
#
#     softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
#     cost = -label * torch.log(softmax_map)
#     cost[label == 0] = 0
#
#     return cost.sum()
#
#
#
# def textureloss(prediction, label, mask_radius,):
#     '''
#     The texture suppression loss that smooths the texture regions.
#     '''
#     filt1 = torch.ones(1, 1, 3, 3)
#     filt1.requires_grad = False
#     filt1 = filt1.to(device)
#     filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
#     filt2.requires_grad = False
#     filt2 = filt2.to(device)
#
#     pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
#     label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)
#
#     mask = 1 - torch.gt(label_sums, 0).float()
#
#     loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
#     loss[mask == 0] = 0
#
#     return torch.sum(loss)
#
#
# def totalloss(prediction, label, l_weight=[0.,0.],):
#     # tracingLoss
#     tex_factor,bdr_factor = l_weight
#     balanced_w = 1.1
#     label = label.float()
#     prediction = prediction.float()
#     with torch.no_grad():
#         mask = label.clone()
#
#         num_positive = torch.sum((mask == 1).float()).float()
#         num_negative = torch.sum((mask == 0).float()).float()
#         beta = num_negative / (num_positive + num_negative)
#         mask[mask == 1] = beta
#         mask[mask == 0] = balanced_w * (1 - beta)
#         mask[mask == 2] = 0
#     prediction = torch.sigmoid(prediction)
#     # print('bce')
#     cost = torch.sum(torch.nn.functional.binary_cross_entropy(
#         prediction.float(), label.float(), weight=mask,  reduction='none'))
#     label_w = (label != 0).float()
#     # print('tex')
#     textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, )
#     bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, )
#
#     return cost + bdr_factor * bdrcost + tex_factor * textcost
