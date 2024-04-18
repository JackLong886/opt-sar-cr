import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataloader import CustomDataset
from network import DSen2CRModel
from loss import CARL
import numpy as np
import random
import para
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    loss_fn = CARL()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 随机种子设置日志
    random.seed(para.random_seed_general)
    np.random.seed(para.random_seed_general)
    torch.manual_seed(para.random_seed_general)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(para.random_seed_general)
    logger.info("Random seeds set.")

    # 模型初始化日志
    model = DSen2CRModel(
        input_shape_opt=(para.in_channels_opt, para.crop_size, para.crop_size),
        input_shape_sar=(para.in_channels_sar, para.crop_size, para.crop_size),
        feature_size=para.feature_size,
        num_layers=para.num_layers,
        include_sar_input=para.include_sar_input,
        use_cloud_mask=para.use_cloud_mask
    )
    model.to(device)
    logger.info("Model initialized and moved to device: %s", device)

    optimizer = optim.Adam(model.parameters(), lr=para.lr, betas=para.betas, eps=para.eps)
    dataset = CustomDataset(para.img_dir, para.gt_dir, para.mask_dir, para.sar_dir, para.trans, para.sar_trans)
    train_loader = DataLoader(dataset, batch_size=para.batch_size, shuffle=para.shuffle_train,
                              num_workers=para.num_workers)
    logger.info("Start Training...")
    for epoch in range(para.num_epochs):
        total_loss = 0.0
        for item, label, sar in train_loader:
            item = item.to(device)
            label = label.to(device)
            sar = sar.to(device)
            outputs = model(item, sar)
            loss = loss_fn(outputs, label)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')

        torch.save(model.state_dict(), f'checkpoint.pth')
        logger.info("Model state dictionary saved.")

    logger.info("Training completed.")
