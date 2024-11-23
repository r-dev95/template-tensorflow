"""This is the module that makes TFRecord data.
"""  # noqa: INP001

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Any

import keras

from lib.common.decorator import process_time
from lib.common.define import ParamKey, ParamLog
from lib.common.log import SetLogging
from lib.data.base import write_exmaple

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def make_tfrecord(args: list[Callable, str]) -> None:
    """Makes TFRecord data.

    *   Make TFRecord data loading data from the following function.

        *   ``keras.datasets.mnist.load_data()``
        *   ``keras.datasets.fashion_mnist.load_data``
        *   ``keras.datasets.cifar10.load_data``
        *   ``keras.datasets.cifar100.load_data``

    Args:
        args (list[Callable, str]): keras's dataset loader and save directory path.

    .. code-block:: python

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        # x_train.shape == (60000, 28, 28)
        # y_train.shape == (60000,)
        # x_test.shape  == (10000, 28, 28)
        # y_test.shape  == (10000,)

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        # x_train.shape == (50000, 32, 32, 3)
        # y_train.shape == (50000, 1)
        # x_test.shape  == (10000, 32, 32, 3)
        # y_test.shape  == (10000, 1)
    """
    load_data, dpath = args

    fpath_train = Path(dpath, 'train', 'train.tfr')
    fpath_test = Path(dpath, 'test', 'test.tfr')
    (x_train, y_train), (x_test, y_test) = load_data()
    # training data.
    LOGGER.info('train data:')
    fpath_train.parent.mkdir(parents=True, exist_ok=True)
    write_exmaple(fpath=fpath_train, inputs=x_train, labels=y_train)
    # test data.
    LOGGER.info('test data:')
    fpath_test.parent.mkdir(parents=True, exist_ok=True)
    write_exmaple(fpath=fpath_test, inputs=x_test, labels=y_test)


@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> None:
    """main.

    Args:
        params (dict[str, Any]): parameters.
    """
    func = {
        'mnist': keras.datasets.mnist.load_data,
        'fashion_mnist': keras.datasets.fashion_mnist.load_data,
        'cifar10': keras.datasets.cifar10.load_data,
        'cifar100': keras.datasets.cifar100.load_data,
    }

    if 'all' in params[K.DATA]:
        params[K.DATA] = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    args = [[func[kind], Path(params[K.RESULT], kind)] for kind in params[K.DATA]]
    with ProcessPoolExecutor(max_workers=params['max_workers']) as executer:
        executer.map(make_tfrecord, args)


def set_params() -> dict[str, Any]:
    """Sets the command line arguments.

    Returns:
        dict[str, Any]: parameters.
    """
    # set the command line arguments.
    parser = argparse.ArgumentParser()
    # log level (idx=0: stream handler, idx=1: file handler)
    # (DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50)
    choices = [10, 20, 30, 40, 50]
    parser.add_argument('--level', default=[20, 20], type=int, nargs=2, choices=choices)
    # directory path (data save)
    parser.add_argument('--result', default='', type=str)
    # data
    choices = ['all', 'mnist', 'fashion_mnist', 'cifar10', 'cifar100']
    parser.add_argument('--data', default='', type=str, nargs='+', choices=choices)
    # max workers
    parser.add_argument('--max_workers', default=8, type=int)

    params = vars(parser.parse_args())

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if K.RESULT in params:
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)