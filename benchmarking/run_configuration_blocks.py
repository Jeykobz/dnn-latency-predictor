import sys
import torch
import blocks
import os
from utils import *

# Extract the passed arguments
block_name = sys.argv[1]
input_c = int(sys.argv[2])
input_h = int(sys.argv[3])
input_w = int(sys.argv[4])
batch_size = int(sys.argv[5])
input = tuple((input_c, input_h, input_w))
device = sys.argv[6]

if block_name == 'BasicBlock7':
    block = blocks.BasicBlock7()
if block_name == 'Block01':
    block = blocks.Block01()
if block_name == 'Bottleneck1':
    block = blocks.Bottleneck1()
if block_name == 'Bottleneck4':
    block = blocks.Bottleneck4()
if block_name == 'Bottleneck5':
    block = blocks.Bottleneck5()
if block_name == 'Conv2d_3x3':
    block = blocks.Conv2d_3x3()
if block_name == 'Bottleneck8':
    block = blocks.Bottleneck8()
if block_name == 'InvertedResidual3':
    block = blocks.InvertedResidual3()
if block_name == 'ResnetBottleneck8':
    block = blocks.ResnetBottleneck8()
if block_name == 'MBConv':
    block = blocks.MBConv()
if block_name == 'ResBottleneckBlock3':
    block = blocks.ResBottleneckBlock3()
if block_name == 'ResnetBottleneck4':
    block = blocks.ResnetBottleneck4()
if block_name == 'Bottleneck9':
    block = blocks.Bottleneck9()
if block_name == 'ResBottleneckBlock1':
    block = blocks.ResBottleneckBlock1()
if block_name == 'InvertedResidual2':
    block = blocks.InvertedResidual2()

# run actual benchmarks
if device == 'gpu':
    block.cuda()
path = '/benchmarks/' + block_name + '_' + str(input) + '_' + str(batch_size)
model_benchmarks = ModelBenchmarks(block, input, device, batch_size, path)
print(model_benchmarks._cuda_error)
torch.cuda.synchronize()