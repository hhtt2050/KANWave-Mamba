import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score


def compute_single_image_stats(pred, label, num_classes, ignore_index=255):
    """
    计算单张图像的 Intersect, Union, Pred, Label 统计量（对应论文中的 `intersect_and_union`）
    Args:
        pred:   [H, W] 预测标签（0~num_classes-1）
        label:  [H, W] 真实标签（0~num_classes-1）
        num_classes: 类别数
        ignore_index: 忽略的像素索引
    Returns:
        area_intersect: [num_classes] 每个类别的交集像素数
        area_union:     [num_classes] 每个类别的并集像素数
        area_pred:      [num_classes] 每个类别的预测像素数
        area_label:     [num_classes] 每个类别的真实像素数
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    # 忽略指定像素
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    # 计算交集 (pred == label 的像素)
    intersect = pred[pred == label]
    # 统计直方图: 0~num_classes 每个区间的计数
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))

    # 统计预测和真实标签的直方图
    area_pred, _ = np.histogram(pred, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))

    # 并集 = 预测 + 真实 - 交集
    area_union = area_pred + area_label - area_intersect

    return area_intersect, area_union, area_pred, area_label

def compute_mIoU(preds, labels, num_classes, ignore_index=255):
    # 计算全局统计量
    total_intersect, total_union, _, _ = _compute_global_stats(preds, labels, num_classes, ignore_index)
    # 计算 mIoU
    iou_per_class = total_intersect / (total_union + 1e-10)
    return np.nanmean(iou_per_class)


def compute_dice(preds, labels, num_classes, ignore_index=255):
    # 计算全局统计量
    total_intersect, _, total_pred, total_label = _compute_global_stats(preds, labels, num_classes, ignore_index)
    # 计算每个类别的 Dice（包括背景）
    dice_per_class = 2 * total_intersect / (total_pred + total_label + 1e-10)
    # ------------------ 修改点：排除背景类别（类别0） ------------------
    # 假设背景是第一个类别（索引0），只保留其他类别的 Dice
    dice_per_class = dice_per_class[1:]
    # 计算非背景类别的平均 Dice（忽略可能的NaN值）
    return np.nanmean(dice_per_class)


def _compute_global_stats(preds, labels, num_classes, ignore_index=255):
    # 初始化累积统计量
    total_intersect = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_pred = np.zeros(num_classes)
    total_label = np.zeros(num_classes)
    # 遍历图像累积统计
    for pred, label in zip(preds, labels):
        area_intersect, area_union, area_pred, area_label = compute_single_image_stats(
            pred, label, num_classes, ignore_index
        )
        total_intersect += area_intersect
        total_union += area_union
        total_pred += area_pred
        total_label += area_label
    return total_intersect, total_union, total_pred, total_label


############################# 以下是保留你原有的指标函数 #############################
def compute_precision(pred, label, num_classes=2):
    """计算精确度（Precision）"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    pred = pred.flatten()
    label = label.flatten()
    return precision_score(label, pred, average='binary')


def compute_recall(pred, label, num_classes=2):
    """计算召回率（Recall）"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    pred = pred.flatten()
    label = label.flatten()
    return recall_score(label, pred, average='binary')


def compute_pixel_accuracy(pred, label):
    """计算像素级准确率"""
    correct = (pred == label).sum().float()
    total = label.numel()
    return correct / total


