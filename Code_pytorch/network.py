import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, feature_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out += identity
        out = self.relu(out)
        return out


class DSen2CRModel(nn.Module):
    def __init__(self, input_shape_opt, input_shape_sar, feature_size=256, num_layers=32,
                 include_sar_input=True, use_cloud_mask=False):
        super(DSen2CRModel, self).__init__()
        self.include_sar_input = include_sar_input
        self.use_cloud_mask = use_cloud_mask
        input_channels = input_shape_opt[0] + input_shape_sar[0] if include_sar_input else input_shape_opt[0]
        self.initial_conv = nn.Conv2d(input_channels, feature_size, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(feature_size) for _ in range(num_layers)])
        self.final_conv = nn.Conv2d(feature_size, input_shape_opt[0], kernel_size=3, padding=1)

    def forward(self, input_opt, input_sar=None):
        if self.include_sar_input and input_sar is not None:
            x = torch.cat([input_opt, input_sar], dim=1)
        else:
            x = input_opt

        x = F.relu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = x + input_opt

        if self.use_cloud_mask:
            # Implement the cloud mask logic if necessary
            pass

        return x


if __name__ == '__main__':
    from thop import profile
    # Example of how to instantiate and use the model
    input_shape_opt = (3, 256, 256)  # Example shape, (channels, height, width)
    input_shape_sar = (2, 256, 256)  # Example shape

    model = DSen2CRModel(input_shape_opt, input_shape_sar)
    totol_para = 0
    for name, parameter in model.named_parameters():
        totol_para += parameter.numel()
    print(f'total parameters: {totol_para / 1024 / 1024:.2f}')

    # # Example input tensorspip install thop
    # input_opt = torch.randn(1, *input_shape_opt)
    # input_sar = torch.randn(1, *input_shape_sar)
    # output = model(input_opt, input_sar)

    print()
