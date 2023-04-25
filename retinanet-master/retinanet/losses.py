import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    # b框的面积
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # 每个a框和b框相交部分的宽
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    # 每个a框和b框相交部分的高
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    # a框与b框的并集
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    # classifications 网络预测类别信息
    # regressions 网络预测回归信息
    # anchors 先验框
    # 真实标注
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]       # anchors的形状是 [1, 每层anchor数量之和 , 4]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            # 获取分类和回归预测结果
            classification = classifications[j, :, :]    # classifications的shape [batch,所有anchor的数量，分类数]
            # print(classification.shape)
            regression = regressions[j, :, :]
            # print(regression.shape)
            # 获得真实标注框信息
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # print(bbox_annotation)

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)   # 将输入张量每个元素的值压缩到一个区间再返回新张量

            if bbox_annotation.shape[0] == 0:
                # 只有负样本时只计算分类focal loss
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue
            # 每个anchor 与 每个标注的真实框的iou
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations
            # Iou_max 每个a框对应的所有b框的最大交并比
            # Iou_argmax 每个先验框对应真实框的索引
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1   # classification 的shape[anchor总数，分类数]

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0            # anchor最大iou小于0.4的置为0,背景

            # 返回bool索引
            positive_indices = torch.ge(IoU_max, 0.5)         # 找出最大iou大于0.5的anchor索引
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]   # 获得每个anchor匹配对应的真实框标注

            # a首先将正样本的类别全部设置为0， 然后在将正样本的类别设置为1， 方便后续的CrossEntropy计算
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            # 对于是物体的使用 1 - pt，对于背景的使用pt
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # alpha_factor * (1 - p)^y
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                # 过滤掉那些值为-1的值
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                # 正样本anchor的w, h, ctr_x, ctr_y
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # 对应真实框的w, h, ctr_x, ctr_y
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi        # 中心点的偏移量dx
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi       # 中心点的偏移量dy
                targets_dw = torch.log(gt_widths / anchor_widths_pi)                # w的log偏移量，降低大框产生的影响
                targets_dh = torch.log(gt_heights / anchor_heights_pi)              # h的log偏移量
                # print(torch.Tensor(targets_dh).detach().numpy())

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    # 对偏移量进行归一化
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # print(torch.Tensor(regression_diff).detach().numpy())

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        # print(torch.Tensor(targets))
        # print(torch.Tensor(regression_losses).detach().numpy())
        # print(torch.Tensor(classification_losses).detach().numpy())
        # print(torch.Tensor([item for item in regression_losses]).detach().numpy())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


