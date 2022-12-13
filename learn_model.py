import subprocess
import os
import sys

# CPU or GPU
try:
    device = sys.argv[1]
    
    if device != 'cpu' and device != 'gpu':
        print('ERROR: NO DEVICE SPECIFIED.\n'
              'Please specify wheather you want to use your CPU or GPU by passing cpu or gpu as parameter.')
    else:
        # Benchmarking
        path = os.getcwd() + '/benchmarking/run_benchmarks_blocks.py'
        run_benchmark = subprocess.run(['python3', path, device])

        # Learn parameters
        path = os.getcwd() + '/benchmarking/learn_parameters.py'
        print('\nLearning prediction model...')
        model = subprocess.run(['python3', path])

except IndexError:
    print('ERROR: NO DEVICE SPECIFIED.\nPlease specify if you want to use your CPU or GPU by passing \'cpu\' or \'gpu\' as parameter.')
