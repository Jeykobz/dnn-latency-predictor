import os
import subprocess
from utils import *
import sys

# GPU or CPU
device = sys.argv[1]

# considered models and inputs
blocks = ['BasicBlock7', 'Conv2d_3x3', 'Bottleneck4', 'ResBottleneckBlock3', 'InvertedResidual3', 'Bottleneck1',
          'Bottleneck9', 'InvertedResidual2', 'MBConv']
input_dims = [[256, 14, 14], [3, 224, 224], [256, 56, 56], [224, 28, 28], [24, 56, 56], [64, 56, 56], [1024, 14, 14],
              [16, 112, 112], [40, 28, 28]]
batch_sizes = [1, 3, 5, 7, 10, 15, 20, 50, 70, 90, 150, 200, 300]
cuda_error = False

# run each block for each batch size or until cuda runtime error occurs
print("Running benchmarks...")
for current_block in range(len(blocks)):
    block = blocks[current_block]
    current_batch_idx = 0
    cuda_error = False
    print('Benchmarking ' + block + ' with input size ' + str(input_dims[current_block]) + '...')

    while not cuda_error and current_batch_idx < len(batch_sizes):
        current_batch_size = batch_sizes[current_batch_idx]

        path = os.getcwd() + '/benchmarking/run_configuration_blocks.py'
        conf_run = subprocess.run(
            ['python3', path, block, str(input_dims[current_block][0]),
             str(input_dims[current_block][1]), str(input_dims[current_block][2]),
             str(current_batch_size), device], capture_output=True)

        if 'False' not in conf_run.stdout.decode('utf-8'):
            cuda_error = True
        else:
            current_batch_idx += 1

print("Benchmarking finished.")
