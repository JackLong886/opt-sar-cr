from torchvision import transforms

model_name = 'DSen2-CR_001'
img_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloud'
gt_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\cloudless'
mask_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\mask'
sar_dir = r'C:\Users\ROG\Desktop\TMP\cr_ds\crop\sar'

initial_epoch = 0  # start at epoch number
num_epochs = 1000
batch_size = 4  # training batch size to distribute over GPUs

eps = 1e-8
in_channels_opt = 3
in_channels_sar = 1

data_mean, data_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
sar_mean, sar_std = 0.5, 0.5
crop_size = 256
trans = transforms.Compose([
    transforms.Resize((crop_size, crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])
sar_trans = transforms.Compose([
    transforms.Resize((crop_size, crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=sar_mean, std=sar_std)
])

# model parameters
num_layers = 16  # B value in paper
feature_size = 256  # F value in paper
include_sar_input = True
use_cloud_mask = True

shuffle_train = True  # shuffle images at training time
data_augmentation = True  # flip and rotate images randomly for data augmentation
random_crop = True  # crop out a part of the input image randomly



n_gpus = 1  # set number of GPUs
num_workers = 4 * n_gpus
random_seed_general = 42

lr = 7e-5
betas = (0.9, 0.999)
