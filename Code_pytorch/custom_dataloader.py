import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


def walk4files(img_dir):
    file_list = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            file_list.append(os.path.join(root, name))
    return file_list


class CustomDataset(Dataset):
    def __init__(self, img_dir, gt_dir, mask_dir, sar_dir, trans, sar_trans):
        self.data = walk4files(img_dir)
        self.labels = walk4files(gt_dir)
        self.mask = walk4files(mask_dir)
        self.sar = walk4files(sar_dir)
        assert len(self.data) == len(self.labels) == len(self.sar) == len(self.mask)
        self.transform = trans
        self.sar_trans = sar_trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = Image.open(self.data[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        sar = Image.open(self.sar[idx]).convert('L')


        item = self.transform(item)
        label = self.transform(label)
        mask = transforms.ToTensor()(mask)
        sar = self.sar_trans(sar)

        label = torch.cat((label, mask), dim=0)
        return item, label, sar


if __name__ == '__main__':
    img_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloud'
    gt_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloudless'
    mask_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\mask'
    sar_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\sar'

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    sar_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2)
    ])
    dataset = CustomDataset(img_dir, gt_dir, mask_dir, sar_dir, trans, sar_trans)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for item, label, sar in train_loader:
        print()
