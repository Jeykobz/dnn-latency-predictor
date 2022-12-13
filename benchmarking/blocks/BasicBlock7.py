import torch

class BasicBlock7(torch.nn.Module):
    def __init__(self):
        super(BasicBlock7, self).__init__()
        self.conv1 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch2 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batch3 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = torch.nn.ReLU(inplace=True)

    def forward(self, x1):
        x = self.conv1(x1)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.conv3(x1)
        x = self.batch3(x)
        x = self.relu3(x)
        return x
