import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


def cloud_mean_absolute_error(y_true, y_pred):
    """计算全图像的平均绝对误差（MAE）。"""
    return tf.reduce_mean(tf.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """计算全图像的均方误差（MSE）。"""
    return tf.reduce_mean(tf.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """计算全图像的均方根误差（RMSE）。"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    """计算每个波段的均方根误差（RMSE）。"""
    return tf.reduce_mean(
        tf.sqrt(tf.reduce_mean(tf.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]), axis=[2, 3])))


def cloud_ssim(y_true, y_pred):
    """计算全图像的结构相似性指数（SSIM）。"""
    y_true = y_true[:, 0:13, :, :]
    y_pred = y_pred[:, 0:13, :, :]

    y_true *= 2000
    y_pred *= 2000

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    return tf.reduce_mean(ssim)


def get_sam(y_true, y_predict):
    """计算光谱角映射（SAM）数组。"""
    # 形状 [batch, channels, height, width]，我们计算的是沿channels的维度
    mat = tf.multiply(y_true, y_predict)
    mat = tf.reduce_sum(mat, axis=1)
    mat = tf.divide(mat, tf.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), axis=1)))
    mat = tf.divide(mat, tf.sqrt(tf.reduce_sum(tf.multiply(y_predict, y_predict), axis=1)))
    mat = tf.acos(tf.clip_by_value(mat, -1.0, 1.0))
    return mat


def cloud_mean_sam(y_true, y_predict):
    """计算全图像的光谱角映射（SAM）均值。"""
    mat = get_sam(y_true[:, 0:13, :, :], y_predict[:, 0:13, :, :])
    return tf.reduce_mean(mat)


def cloud_mean_sam_covered(y_true, y_pred):
    """计算被覆盖部分的光谱角映射（SAM）。"""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:13, :, :]
    predicted = y_pred[:, 0:13, :, :]
    if tf.reduce_sum(cloud_cloudshadow_mask) == 0:
        return 0.0
    sam = get_sam(target, predicted)
    sam = tf.expand_dims(sam, axis=1)
    sam = tf.reduce_sum(cloud_cloudshadow_mask * sam) / tf.reduce_sum(cloud_cloudshadow_mask)
    return sam


def cloud_mean_sam_clear(y_true, y_pred):
    """计算清晰部分的光谱角映射（SAM）。"""
    # 假设最后一个通道是掩模层
    clearmask = 1 - y_true[:, -1, :, :]  # 确保这是单个掩模层，而不是带有冒号的切片
    target = y_true[:, :13, :, :]  # 取前13个通道作为目标
    predicted = y_pred[:, :13, :, :]  # 取前13个通道作为预测

    if tf.reduce_sum(clearmask) == 0:
        return 0.0  # 如果清晰掩模的总和为0，说明没有清晰区域，返回0.0

    sam = get_sam(target, predicted)
    sam = tf.expand_dims(sam, axis=1)  # 将sam沿新的通道轴扩展，为了和掩模相乘
    sam = tf.reduce_sum(clearmask * sam) / tf.reduce_sum(clearmask)  # 按掩模加权平均

    return sam


def cloud_psnr(y_true, y_predict):
    """计算全图像的峰值信噪比（PSNR）。"""
    y_true = y_true * 2000
    y_predict = y_predict * 2000
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_predict[:, 0:13, :, :] - y_true[:, 0:13, :, :])))
    return 20.0 * (tf.math.log(10000.0 / rmse) / tf.math.log(10.0))


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """计算清晰部分的平均绝对误差（MAE）。"""
    clearmask = 1 - y_true[:, -1, :, :]  # 确保这是单个掩模层，而不是带有冒号的切片
    clearmask = tf.expand_dims(clearmask, axis=1)

    target = y_true[:, :13, :, :]  # 取前13个通道作为目标
    predicted = y_pred[:, :13, :, :]  # 取前13个通道作为预测
    if tf.reduce_sum(clearmask) == 0:
        return 0.0

    clti = clearmask * tf.abs(predicted - target)
    clti = tf.reduce_sum(clti) / tf.reduce_sum(clearmask * 13)
    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """计算被覆盖部分的平均绝对误差（MAE）。"""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    target = y_true[:, 0:13, :, :]
    if tf.reduce_sum(cloud_cloudshadow_mask) == 0:
        return 0.0
    ccmaec = cloud_cloudshadow_mask * tf.abs(predicted - target)
    ccmaec = tf.reduce_sum(ccmaec) / (tf.reduce_sum(cloud_cloudshadow_mask) * 13)
    return ccmaec


def carl_error(y_true, y_pred):
    """计算云适应性正则化损失（CARL）。"""
    # 假设最后一个通道是云/阴影掩模
    cloud_cloudshadow_mask = y_true[:, -1, :, :]
    clearmask = 1 - cloud_cloudshadow_mask  # 计算清晰掩模
    cloud_cloudshadow_mask = tf.expand_dims(cloud_cloudshadow_mask, axis=1)
    clearmask = tf.expand_dims(clearmask, axis=1)

    predicted = y_pred[:, :13, :, :]
    target = y_true[:, :13, :, :]

    # 计算不同部分的MAE
    clear_mae = clearmask * tf.abs(predicted - target)
    cloud_mae = cloud_cloudshadow_mask * tf.abs(predicted - target)
    total_mae = tf.abs(predicted - target)

    # 结合所有部分的损失
    cscmae = tf.reduce_mean(clear_mae + cloud_mae) + 1.0 * tf.reduce_mean(total_mae)
    return cscmae


if __name__ == '__main__':
    np.random.seed(42)
    batch_size = 5
    height, width = 64, 64
    channels = 13

    # 创建模拟的真实数据和预测数据
    y_true = np.random.random((batch_size, channels, height, width)).astype(np.float32)
    y_pred = np.random.random((batch_size, channels, height, width)).astype(np.float32)

    # 创建一个额外的云/阴影掩模层
    cloud_mask = np.random.randint(0, 2, (batch_size, 1, height, width)).astype(np.float32)

    # 将掩模层附加到 y_true 中模拟实际情况
    y_true_with_mask = np.concatenate([y_true, cloud_mask], axis=1)

    # 将 NumPy 数组转换为 TensorFlow 张量
    y_true_tf = tf.constant(y_true)
    y_pred_tf = tf.constant(y_pred)
    y_true_with_mask_tf = tf.constant(y_true_with_mask)

    # 测试各个函数
    mae = cloud_mean_absolute_error(y_true_tf, y_pred_tf)
    mse = cloud_mean_squared_error(y_true_tf, y_pred_tf)
    rmse = cloud_root_mean_squared_error(y_true_tf, y_pred_tf)
    bw_rmse = cloud_bandwise_root_mean_squared_error(y_true_tf, y_pred_tf)
    ssim = cloud_ssim(y_true_tf, y_pred_tf)
    sam_mean = cloud_mean_sam(y_true_tf, y_pred_tf)

    sam_covered = cloud_mean_sam_covered(y_true_with_mask_tf, y_pred_tf)
    sam_clear = cloud_mean_sam_clear(y_true_with_mask_tf, y_pred_tf)
    psnr = cloud_psnr(y_true_tf, y_pred_tf)
    mae_clear = cloud_mean_absolute_error_clear(y_true_with_mask_tf, y_pred_tf)
    mae_covered = cloud_mean_absolute_error_covered(y_true_with_mask_tf, y_pred_tf)
    carl_loss = carl_error(y_true_with_mask_tf, y_pred_tf)

    # 输出结果
    print("Mean Absolute Error:", mae.numpy())
    print("Mean Squared Error:", mse.numpy())
    print("Root Mean Squared Error:", rmse.numpy())
    print("Bandwise Root Mean Squared Error:", bw_rmse.numpy())
    print("Structural Similarity Index:", ssim.numpy())
    print("Mean SAM:", sam_mean.numpy())
    print("Mean SAM Covered:", sam_covered.numpy())
    print("Mean SAM Clear:", sam_clear.numpy())
    print("PSNR:", psnr.numpy())
    print("Mean Absolute Error Clear:", mae_clear.numpy())
    print("Mean Absolute Error Covered:", mae_covered.numpy())
    print("CARL Loss:", carl_loss.numpy())
