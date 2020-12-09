import torch
from torch import nn
from torch.nn import functional as F


# from https://arxiv.org/pdf/1602.07261.pdf
class InceptionResnetV1(nn.Module):
    def __init__(
        self,
        dropout_prob=0.6,
        scale_inception_a=0.17,
        scale_inception_b=0.10,
        scale_inception_c=0.20,
    ):
        super().__init__()

        # layers
        self.stem = Stem()
        self.inception_a_blocks = nn.Sequential(
            *[InceptionA(scale=scale_inception_a) for _ in range(5)]
        )
        self.reduction_a = ReductionA()
        self.inception_b_blocks = nn.Sequential(
            *[InceptionB(scale=scale_inception_b) for _ in range(10)]
        )
        self.reduction_b = ReductionB()
        self.inception_c_blocks = nn.Sequential(
            *[InceptionC(scale=scale_inception_c) for _ in range(5)]
        )
        self.inception_c = InceptionC(apply_relu=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear = nn.Linear(1792, 512, bias=False)
        self.bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x_1 = self.stem(x)
        x_2 = self.inception_a_blocks(x_1)
        x_3 = self.reduction_a(x_2)
        x_4 = self.inception_b_blocks(x_3)
        x_5 = self.reduction_b(x_4)
        x_6 = self.inception_c_blocks(x_5)
        x_7 = self.inception_c(x_6)
        x_8 = self.avg_pool(x_7)
        x_9 = self.dropout(x_8)
        x_10 = self.linear(x_9.view(x_9.shape[0], -1))
        x_11 = self.bn(x_10)
        out = F.normalize(x_11, p=2, dim=1)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,  # value from tensorflow implementation
            momentum=0.1,  # default value
            affine=True,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.conv(x)
        x_2 = self.bn(x_1)
        out = self.relu(x_2)
        return out


class Stem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv2d_4 = BasicConv2d(
            64, 80, kernel_size=1, stride=1
        )  # no padding needed for 1x1 conv
        self.conv2d_5 = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_6 = BasicConv2d(192, 256, kernel_size=3, stride=2)

    def forward(self, x):
        x_1 = self.conv2d_1(x)
        x_2 = self.conv2d_2(x_1)
        x_3 = self.conv2d_3(x_2)
        x_4 = self.maxpool(x_3)
        x_5 = self.conv2d_4(x_4)
        x_6 = self.conv2d_5(x_5)
        out = self.conv2d_6(x_6)
        return out


class InceptionA(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch_1 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch_2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.branch_3 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv2d = BasicConv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_a = self.branch_1(x)
        x_b = self.branch_2(x)
        x_c = self.branch_3(x)

        x_1 = torch.cat((x_a, x_b, x_c), 1)
        x_2 = self.conv2d(x_1)
        x_3 = x_2 * self.scale + x
        out = self.relu(x_3)
        return out


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch_1 = nn.MaxPool2d(3, stride=2)

        self.branch_2 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch_3 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )

    def forward(self, x):
        x_a = self.branch_1(x)
        x_b = self.branch_2(x)
        x_c = self.branch_3(x)

        out = torch.cat((x_a, x_b, x_c), 1)
        return out


class InceptionB(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch_1 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch_2 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_a = self.branch_1(x)
        x_b = self.branch_2(x)

        x_1 = torch.cat((x_a, x_b), 1)
        x_2 = self.conv2d(x_1)
        x_3 = x_2 * self.scale + x
        out = self.relu(x_3)
        return out


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch_1 = nn.MaxPool2d(3, stride=2)

        self.branch_2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch_3 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch_4 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

    def forward(self, x):
        x_a = self.branch_1(x)
        x_b = self.branch_2(x)
        x_c = self.branch_3(x)
        x_d = self.branch_4(x)

        out = torch.cat((x_a, x_b, x_c, x_d), 1)
        return out


class InceptionC(nn.Module):
    def __init__(self, scale=1.0, apply_relu=True):
        super().__init__()

        self.scale = scale
        self.apply_relu = apply_relu

        self.branch_1 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch_2 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if self.apply_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x_a = self.branch_1(x)
        x_b = self.branch_2(x)

        x_1 = torch.cat((x_a, x_b), 1)
        x_2 = self.conv2d(x_1)
        out = x_2 * self.scale + x
        if self.apply_relu:
            out = self.relu(out)
        return out
