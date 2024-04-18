from PIL import Image
import os

img_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloud'
gt_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloudless'
mask_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\mask'
sar_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\sar'

from custom_dataloader import walk4files

masks = walk4files(mask_dir)
for mask in masks:
    m_data = Image.open(mask).convert('L')
    if max(list(m_data.getdata())) == 0:
        basename = os.path.basename(mask)
        os.remove(mask)
        os.remove(os.path.join(gt_dir, basename))
        os.remove(os.path.join(sar_dir, basename))
        os.remove(os.path.join(img_dir, basename))

print()