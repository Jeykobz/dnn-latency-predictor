import torch

class MBConv(torch.nn.Module):
    def __init__(self):
        super(MBConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch1 = torch.nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu1 = torch.nn.SiLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batch2 = torch.nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu2 = torch.nn.SiLU(inplace=True)
        self.adaptive3 = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3 = torch.nn.Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu3 = torch.nn.SiLU(inplace=True)
        self.conv3_1 = torch.nn.Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batch4 = torch.nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.silu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.silu2(x)
        x = self.adaptive3(x)
        x = self.conv3(x)
        x = self.silu3(x)
        x = self.conv3_1(x)
        x = self.sigmoid3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        return x
