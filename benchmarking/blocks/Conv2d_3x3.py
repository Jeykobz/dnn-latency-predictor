import torch

class Conv2d_3x3(torch.nn.Module):
    def __init__(self):
        super(Conv2d_3x3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.batch2 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch3 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv4 = torch.nn.Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch4 = torch.nn.BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = torch.nn.Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.batch5 = torch.nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.maxpool5(x)
        return x
