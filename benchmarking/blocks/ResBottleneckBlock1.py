import torch


class ResBottleneckBlock1(torch.nn.Module):
    def __init__(self):
        super(ResBottleneckBlock1, self).__init__()
        self.conv1 = torch.nn.Conv2d(32, 224, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(32, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch2 = torch.nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
        self.batch3 = torch.nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.adapt4 = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv4 = torch.nn.Conv2d(224, 8, kernel_size=(1, 1), stride=(1, 1))
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.conv5 = torch.nn.Conv2d(8, 224, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = torch.nn.Sigmoid()
        self.conv6 = torch.nn.Conv2d(224, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch6 = torch.nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = torch.nn.ReLU(inplace=True)


    def forward(self, x1):
        x = self.conv1(x1)
        x = self.batch1(x)
        x = self.conv2(x1)
        x = self.batch2(x)
        x2 = self.relu2(x)
        x = self.conv3(x2)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.adapt4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.sigmoid5(x)
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        return x
