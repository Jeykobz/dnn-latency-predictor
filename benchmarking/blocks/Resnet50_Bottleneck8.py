import torch

class ResnetBottleneck8(torch.nn.Module):
    def __init__(self):
        super(ResnetBottleneck8, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch2 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch3 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batch4 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = torch.nn.ReLU(inplace=True)

    def forward(self, x1):
        x = self.conv1(x1)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.conv4(x1)
        x = self.batch4(x)
        x = self.relu4(x)
        return x
