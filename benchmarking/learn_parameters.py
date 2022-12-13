import os
import json
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit, minimize
from numpyencoder import NumpyEncoder


class LearnModel(object):
    def __init__(self):
        # loading all the benchmark json files contained in /Benchmarks
        benchmark_path = os.getcwd() + '/benchmarks'
        self._benchmark_files = [f for f in listdir(benchmark_path) if isfile(join(benchmark_path, f))]

        self._latencies = []
        self._inputs = []
        self._acts = []
        self._flops = []
        self._block_name = []
        self._x_axis = []
        self._breakpoint = 0
        self.check_training_data()

    def check_training_data(self):
        # check if directory contains non-json files
        dirty_directory = False
        for benchmark_path in self._benchmark_files:
            if '.json' not in benchmark_path:
                dirty_directory = True
        if dirty_directory:
            print(
                'Error: The /benchmarks directory contains non-json files.\n'
                'Please remove them from the directory and start the process again.')
        else:
            self.load_benchmarks()
            self.learn_model()

    def load_benchmarks(self):
        """
        Extracts all benchmarks from the /Benchmarks directory and checks for possible errors.
        """
        # check quantity of benchmarks
        if len(self._benchmark_files) == 0:
            print(
                'ERROR: No benchmarks in /benchmarks found.'
                '\nPlease add all generated benchmarks to the /Benchmarks directory and start the process again.')
        else:
            if len(self._benchmark_files) < 10:
                print('The predictor may be very inaccurate due to insufficient training data.')

            for bench_file in self._benchmark_files:
                # load benchmarks from json file
                bench_path = os.getcwd() + '/benchmarks/' + bench_file
                with open(bench_path, 'r') as json_target:
                    benchmarks = json.load(json_target)

                self._latencies.append(benchmarks[0]['latency'])
                self._inputs.append(benchmarks[0]['input'])
                self._acts.append(benchmarks[0]['acts'])
                self._flops.append(benchmarks[0]['flops'])
                self._block_name.append(bench_file)

    def objective_fair(self, X, a, b, c, d):
        """
        Function to be optimized to find the best fitting parameters a, b, c and d.
        :param X: List containing the extracted benchmark data
        :param a: Parameter that controls the influence of the number of activations on the prediction
        :param b: Parameter that controls the influence of the Size of the inputs on the prediction
        :param c: Parameter that controls the influence of the number of FLOPS on the prediction
        :param d: Learned bias
        :return: Result of the function to be minimized by curve_fit()
        """
        inp, activations, flops = X
        return abs(a) * activations + abs(b) * inp + abs(c) * flops + abs(d)

    def get_fair_model(self, X, Y):
        """
        Finds an optimal model by learning the parameters a, b, c, d that fit the data.
        :param X: List of activations, input sizes and FLOPS of each block
        :param Y: List of latencies of each block
        :return: Learned parameters
        """
        (a, b, c, d), _ = curve_fit(self.objective_fair, X, Y)
        return abs(a), abs(b), abs(c), abs(d)

    def learn_model(self):
        X = [self._inputs, self._acts, self._flops]
        Y = self._latencies
        a, b, c, d = self.get_fair_model(X, Y)
        breakpoint, slope, offset = self.find_breakpoint(a, b, c, d)

        result_path = os.getcwd() + '/predictor.json'
        pred_params = [
            {
                'a': a, 'b': b, 'c': c, 'd': d,
                'breakpoint': breakpoint, 'slope': slope,
                'offset': offset
            }
        ]
        with open(result_path, 'w') as json_target:
            json.dump(pred_params, json_target, cls=NumpyEncoder)

        print(f'Success! The prediction model was successfully learned.\na: {a}, b: {b}, c: {c}, d: {d}. \n'
              f'Optimal break point: x = {breakpoint} with slope: s = {slope} and offset = {offset}')

    def rel_error(self, slope_offset):
        """
        Function to be optimized to find the optimal breakpoint and slope for the
        lower latency regions. Lower latencies are otherwise overestimated.

        It calculates the percentage deviation of the predictions at a given breakpoint currently under consideration.
        :param s: Slope parameter to be learned.
        :return: Calculated relative error
        """
        s = slope_offset[0]
        offset = slope_offset[1]
        err = 0
        for x in range(len(self._x_axis)):
            if self._x_axis[x] < self._breakpoint:
                err += np.abs((self._latencies[x] - (s * self._x_axis[x] + offset)) / self._latencies[x]) * 100
            else:
                err += np.abs((self._latencies[x] - self._x_axis[x]) / self._latencies[x]) * 100

        return err

    def minimize_rel_err(self):
        """
        Finds an optimal slope parameter s for the given breakpoint.
        :return: Learned slope parameter s
        """
        result = scipy.optimize.minimize(self.rel_error, [1, 1])
        return result.x

    def find_breakpoint(self, a, b, c, d):
        """
        Regions with lower latencies tend to be overestimated when only a single regression model is used.
        Benchmarks with higher latencies dominate benchmarks with lower latencies when learning the regression model.
        To solve this problem, we need to modify the regression model for regions with lower latencies.

        To do this, we need to find a breakpoint that indicates the optimal point to
        which the modification of the regression model should go.
        The idea is to iterate over the x-axis with a small step size.
        At each step, the current x-value is considered the current breakpoint.
        We evaluate the quality of this breakpoint by examining how the overall relative error changes
        as we apply the breakpoint currently under consideration.
        If the current breakpoint results in a smaller relative error,
        then this breakpoint replaces the previous optimal breakpoint.

        To save time and computational effort, the entire x-axis is not considered, but only up to a certain extent.

        :param a: Parameter that controls the influence of the number of activations on the prediction
        :param b: Parameter that controls the influence of the Size of the inputs on the prediction
        :param c: Parameter that controls the influence of the number of FLOPS on the prediction
        :param d: Learned bias
        :return: Learned optimal breakpoint and slope
        """
        data = {'latencies': self._latencies, 'name': self._block_name,
                'acts': self._acts, 'flops': self._flops, 'inp': self._inputs}
        self._x_axis = []
        for idx in range(len(data['latencies'])):
            self._x_axis.append(a * data['acts'][idx] + b * data['inp'][idx] + c * data['flops'][idx] + d)

        self._breakpoint = d
        first_run = True
        opt_err = 0
        opt_s = 0
        opt_br_point = 0
        opt_offset = 0
        step_size = max(self._x_axis) * 10**-3

        while self._breakpoint < max(self._x_axis) / 10:
            s, offset = self.minimize_rel_err()
            err = 0
            for x in range(len(self._x_axis)):
                if self._x_axis[x] < self._breakpoint:
                    err += np.abs((self._latencies[x] - (s * self._x_axis[x] + offset)) / self._latencies[x]) * 100
                else:
                    err += np.abs((self._latencies[x] - self._x_axis[x]) / self._latencies[x]) * 100

            if not first_run:
                if err < opt_err:
                    opt_err = err
                    opt_s = s
                    opt_offset = offset
                    opt_br_point = self._breakpoint
            else:
                first_run = False
                opt_err = err
                opt_s = s
                opt_offset = offset
                opt_br_point = self._breakpoint

            self._breakpoint += step_size

        return [opt_br_point, opt_s, opt_offset]


LearnModel()
