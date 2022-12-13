import torch

class Block01(torch.nn.Module):
    def __init__(self):
        super(Block01, self).__init__()
        self.conv1 = torch.nn.Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        self.batch2 = torch.nn.BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch3 = torch.nn.BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        return x
