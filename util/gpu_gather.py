# -----------------------------------------------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# -----------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------------------------------------------------

from typing import Callable

from functools import wraps
import torch.distributed as dist
import torch

import pickle
from collections import defaultdict
import numpy as np
from itertools import chain




def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    rank = dist.get_rank()
    device = torch.device('cuda', rank)

    # serialized to a Tensor
    buffer = pickle.dumps(data)

    storage = torch.ByteStorage.from_buffer(buffer)

    tensor = torch.ByteTensor(storage)

    tensor = tensor.to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]

    dist.barrier()

    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


"""
https://github.com/pytorch/ignite/blob/master/ignite/metrics/epoch_metric.py#L137
代码出处


"""


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


def reinit__is_reduced(func: Callable) -> Callable:
    """Helper decorator for distributed configuration.
    See :doc:`metrics` on how to use it.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._is_reduced = False

    wrapper._decorated = True
    return wrapper


class GpuGather:
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.
    .. warning::
        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.
    .. warning::
        Current implementation does not work with distributed computations. Results are not gather across all devices
        and computed results are valid for a single device only.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.
    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    """

    def __init__(self, is_distributed):
        self.meters = defaultdict(list)
        self._is_reduced = False
        self.is_distributed = is_distributed
        self.reset()

    @reinit__is_reduced
    def reset(self) -> None:
        self.meters.clear()

    @reinit__is_reduced
    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (list, np.ndarray))
            if isinstance(v, list):
                self.meters[k].extend(v)
            else:
                self.meters[k].append(v)

    def synchronize_between_processes(self):

        if not self.is_distributed:
            return
        ws = dist.get_world_size()

        dist.barrier()
        if ws > 1 and not self._is_reduced:

            for k, v in self.meters.items():
                self.meters[k] = list(chain(*all_gather(v)))

        self._is_reduced = True

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
