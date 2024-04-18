import numpy as np
import os
import torch
import torch.nn as nn
from pytorch_msssim import ssim

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import torch


def cloud_mean_absolute_error(y_true, y_pred):
    """计算全图像的平均绝对误差（MAE）。"""
    return torch.mean(torch.abs(y_pred - y_true))


def cloud_mean_squared_error(y_true, y_pred):
    """计算全图像的均方误差（MSE）。"""
    return torch.mean((y_pred - y_true) ** 2)


def cloud_root_mean_squared_error(y_true, y_pred):
    """计算全图像的均方根误差（RMSE）。"""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    """计算每个波段的均方根误差（RMSE）。
    y_true 和 y_pred 的形状应为 [batch_size, channels, height, width]。
    """
    # 计算每个波段的均方误差
    mse = torch.mean((y_pred - y_true) ** 2, dim=[2, 3])  # 沿着高和宽计算
    # 计算均方根误差
    rmse = torch.sqrt(mse)
    # 计算所有波段的均值
    bandwise_rmse = torch.mean(rmse, dim=1)  # 沿着通道维度取均值
    return bandwise_rmse


def cloud_ssim(y_true, y_pred, data_range=10000.0):
    """计算全图像的结构相似性指数（SSIM）。"""
    y_true = y_true * 2000
    y_pred = y_pred * 2000

    # 确保输入为 float 类型
    y_true = y_true.float()
    y_pred = y_pred.float()

    return ssim(y_true, y_pred, data_range=data_range)


def get_sam(y_true, y_pred):
    """计算光谱角映射（SAM）数组。"""
    # 计算两个向量的点积
    mat = torch.sum(y_true * y_pred, dim=1)
    # 计算两个向量的模
    norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1))
    norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
    # 计算余弦相似度
    cos_sim = mat / (norm_true * norm_pred)
    # 通过arccos得到角度，确保值在-1.0到1.0之间以防止数值错误
    sam = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
    return sam


def cloud_mean_sam(y_true, y_pred):
    """计算全图像的光谱角映射（SAM）均值。"""
    sam = get_sam(y_true, y_pred)
    return torch.mean(sam)


def cloud_mean_sam_covered(y_true, y_pred):
    """计算被覆盖部分的光谱角映射（SAM）。"""
    cloud_cloudshadow_mask = y_true[:, -1, :, :]
    y_true = y_true[:, :-1, :, :]
    if torch.sum(cloud_cloudshadow_mask) == 0:
        return torch.tensor(0.0)
    sam = get_sam(y_true, y_pred)
    # 使用权重平均，避免除以零
    sam_weighted = cloud_cloudshadow_mask * sam
    return torch.sum(sam_weighted) / torch.sum(cloud_cloudshadow_mask)


def cloud_mean_sam_clear(y_true, y_pred):
    """计算清晰部分的光谱角映射（SAM）。"""
    clearmask = 1 - y_true[:, -1, :, :]
    y_true = y_true[:, :-1, :, :]
    if torch.sum(clearmask) == 0:
        return torch.tensor(0.0)
    sam = get_sam(y_true, y_pred)
    # 使用权重平均，避免除以零
    sam_weighted = clearmask * sam
    return torch.sum(sam_weighted) / torch.sum(clearmask)


def cloud_psnr(y_true, y_pred):
    """计算全图像的峰值信噪比（PSNR）。"""
    y_true = y_true * 2000
    y_pred = y_pred * 2000
    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    max_pixel = 10000.0
    psnr = 20 * torch.log10(max_pixel / rmse)
    return psnr


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """计算清晰部分的平均绝对误差（MAE）。"""
    clear_mask = 1 - y_true[:, -1, :, :]
    clear_mask = clear_mask.unsqueeze(1)

    y_true = y_true[:, :-1, :, :]
    if torch.sum(clear_mask) == 0:
        return torch.tensor(0.0)

    abs_diff = torch.abs(y_pred - y_true)
    clear_mae = clear_mask * abs_diff
    total_clear_mae = torch.sum(clear_mae) / torch.sum(clear_mask)

    return total_clear_mae


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """计算被覆盖部分的平均绝对误差（MAE）。"""
    cloud_cloudshadow_mask = y_true[:, -1, :, :]
    y_true = y_true[:, :-1, :, :]
    if torch.sum(cloud_cloudshadow_mask) == 0:
        return torch.tensor(0.0)

    abs_diff = torch.abs(y_pred - y_true)
    cloud_mae = cloud_cloudshadow_mask * abs_diff
    total_cloud_mae = torch.sum(cloud_mae) / torch.sum(cloud_cloudshadow_mask)

    return total_cloud_mae


def carl_error(y_true, y_pred):
    """计算云适应性正则化损失（CARL）。
       假设 y_true 的最后一个通道是云/阴影掩模。
    """
    cloud_cloudshadow_mask = y_true[:, -1, :, :]
    y_true = y_true[:, :-1, :, :]
    clearmask = 1 - cloud_cloudshadow_mask
    cloud_cloudshadow_mask = cloud_cloudshadow_mask.unsqueeze(1)
    clearmask = clearmask.unsqueeze(1)
    clear_mae = clearmask * torch.abs(y_pred - y_true)
    cloud_mae = cloud_cloudshadow_mask * torch.abs(y_pred - y_true)
    total_mae = torch.abs(y_pred - y_true)

    cscmae = torch.mean(clear_mae + cloud_mae) + 1.0 * torch.mean(total_mae)

    return cscmae


class CARL(nn.Module):
    def __init__(self):
        super(CARL, self).__init__()

    def forward(self, predictions, targets):
        # 计算差异
        return carl_error(targets, predictions)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建合成的真实值和预测值张量，包含一个云掩模通道
    y_true = torch.rand(1, 14, 256, 256, device=device)  # 包含13个通道的真实值和一个掩模通道
    y_pred = torch.rand(1, 13, 256, 256, device=device)  # 13个通道的预测值，无掩模

    # MAE, MSE, RMSE
    mae = cloud_mean_absolute_error(y_true[:, :-1, :, :], y_pred)
    mse = cloud_mean_squared_error(y_true[:, :-1, :, :], y_pred)
    rmse = cloud_root_mean_squared_error(y_true[:, :-1, :, :], y_pred)

    # Bandwise RMSE（未实现，此函数需要修改）
    bandwise_rmse = cloud_bandwise_root_mean_squared_error(y_true[:, :-1, :, :], y_pred)

    # SAM
    mean_sam = cloud_mean_sam(y_true[:, :-1, :, :], y_pred)

    # SAM for covered and clear areas
    mean_sam_covered = cloud_mean_sam_covered(y_true, y_pred)
    mean_sam_clear = cloud_mean_sam_clear(y_true, y_pred)

    # PSNR
    psnr = cloud_psnr(y_true[:, :-1, :, :], y_pred)

    # MAE for clear and covered areas
    mae_clear = cloud_mean_absolute_error_clear(y_true, y_pred)
    mae_covered = cloud_mean_absolute_error_covered(y_true, y_pred)

    # CARL error
    carl = carl_error(y_true, y_pred)

    # Print the results
    print(f"MAE: {mae.item()}")
    print(f"MSE: {mse.item()}")
    print(f"RMSE: {rmse.item()}")
    print("Bandwise RMSE:", bandwise_rmse.item())
    print(f"Mean SAM: {mean_sam.item()}")
    print(f"Mean SAM Covered: {mean_sam_covered.item()}")
    print(f"Mean SAM Clear: {mean_sam_clear.item()}")
    print(f"PSNR: {psnr.item()}")
    print(f"MAE Clear: {mae_clear.item()}")
    print(f"MAE Covered: {mae_covered.item()}")
    print(f"CARL Error: {carl.item()}")
