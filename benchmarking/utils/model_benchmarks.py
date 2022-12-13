from utils.macs import calculate_macs
import torch
import torch.cuda
import torch.nn as nn
import sys
import os
import json
import logging
import numpy as np
from typing import Dict, Tuple, Callable, Any
from numpyencoder import NumpyEncoder
import time


class ModelBenchmarks(object):
    """
    Responsible for benchmarking.
    Modifies the given model to benchmark it. Each layer within
    the new model gets a modified calling function (a hook) to not
    only compute the output also to measure the latency and to
    compute the input size, output size and the number of FLOPs.
    Runs an evaluation on the modified model and thereby
    benchmarks it and stores each benchmark layer by layer.

    Attributes:
        _origin: dictionary filled with all layers to be analyzed
        _model: copy of passed model
        _inp_size: size of inputs the neural network gets:
        [channels, height, width]
    """

    def __init__(
            self, model: nn.Module,
            inp_size: Tuple[int, int, int], device, batch_size=1, json_name='model') -> None:
        """
        Inits model for benchmarking and writes the results into a json file.
        Also checks if a runtime error occurred.

        :param model: passed model to analyze
        :param inp_size: dimensions of input tensor
        :param batch_size: batch size for analysis process
        """
        assert isinstance(model, nn.Module)
        assert isinstance(inp_size, (list, tuple))
        assert len(inp_size) == 3

        self._origin: Dict[Callable[..., Any],
                           Callable[..., Any]] = dict()
        self._model: nn.Module = model
        self._inp_size: Tuple[int, int, int] = inp_size
        self._batch_size = batch_size
        self._layerwise_benchmarks = []
        self._structure = []
        self._cuda_error = False
        self._error = 'No error!'
        self._device = device

        self._latency = 0
        self._acts = 0
        self._inp = 0
        self._flops = 0

        self.analyse()

        if not self._cuda_error:
            benchmark = [{
                'latency': self._latency,
                'acts': self._acts,
                'input': self._inp,
                'flops': self._flops
            }]

            json_name = os.getcwd() + json_name + '_benchmarks.json'
            with open(json_name, 'w') as json_target:
                json.dump(benchmark, json_target, cls=NumpyEncoder)

    def analyse(self) -> None:
        """
        Provides a structured analysis by calling required methods
        and performing an evaluation with randomised input
        """
        self._modify_submodules()

        # distinguish between CPU and GPU
        if self._device == 'gpu':
            rand_input = torch.rand(self._batch_size, *self._inp_size).cuda()
            self._model.cuda()
        else:
            rand_input = torch.rand(self._batch_size, *self._inp_size)
        self._model.eval()
        self._model(rand_input)

    def _modify_submodules(self) -> None:
        """
        Iterates over all layers/modules contained in given model.
        Collects all types of modules to be analysed and
        modifies the calling functions of each layer for analysis.
        _Origin serves as a dictionary filled with all layer
        types to be analysed. When the for loop is executed, for each
        layer types within the model (such as Conv2d, ReLU, ...)
        the calling function is modified once. _origin remembers
        if a given type of module has already been modified.
        """

        def analyse_each_layer(
                layer: nn.Module, *inp_tensor,
                **vars: Any):
            """
            The new calling function for given layer type.
            During evaluation, when calculating the output for given
            input, this modified function (hook) is called instead of the
            normally called functions for calculation.
            It executes the same model run five times and takes the mean latency of all these runs.
            It uses torch.cuda.Event() and torch.cuda.synchronize() to measure the time required to
            compute the output.
            It also sums up all input sizes, output sizes and number of FLOPS to get total numbers
            of the entire model of each feature.

            :param layer: given module to be analysed
            :param inp_tensor: tensor which serves as input for layer
            :param vars: allows to pass further params
            :return: calculated output of layer for a given input
            """
            assert layer.__class__ in self._origin

            try:
                # warm up
                layer_output = self._origin[layer.__class__](
                    layer, *inp_tensor, **vars)

                # distinguish between GPU and CPU
                total_measured_times = []
                if self._device == 'gpu':
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    # five measured run, take average time
                    for i in range(5):
                        start.record()
                        layer_output = self._origin[layer.__class__](
                            layer, *inp_tensor, **vars)
                        end.record()
                        torch.cuda.synchronize()
                        total_measured_times.append(start.elapsed_time(end))
                else:
                    for i in range(5):
                        start = time.time()
                        layer_output = self._origin[layer.__class__](
                            layer, *inp_tensor, **vars)
                        end = time.time()
                        total_measured_times.append((end - start) * 1000)

                if len(inp_tensor) == 1:
                    flops_counted = \
                        calculate_macs(layer, inp_tensor[0], layer_output)
                elif len(inp_tensor) > 1:
                    flops_counted = \
                        calculate_macs(layer, inp_tensor, layer_output)

                input_shape = 0
                if len(inp_tensor) == 1:
                    input_shape += np.prod(inp_tensor[0].shape)
                elif len(inp_tensor) > 1:
                    for dim in inp_tensor:
                        input_shape += np.prod(dim.shape)

                self._latency += np.median(total_measured_times)
                self._flops += flops_counted
                if isinstance(layer, nn.Conv2d):
                    self._acts += np.prod(tuple(layer_output.shape))
                    self._inp += input_shape

                return layer_output

            except RuntimeError as e:
                self._error = e
                self._cuda_error = True

        # Creates a hook for each operator
        for layer in self._model.modules():
            if layer.__class__ not in self._origin and \
                    len(list(layer.children())) == 0:
                self._origin[layer.__class__] = layer.__class__.__call__
                layer.__class__.__call__ = analyse_each_layer