"""This is the module that makes TFRecord data.
"""  # noqa: INP001

import argparse
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from lib.common.decorator import process_time
from lib.common.log import SetLogging
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


#: str: temporary path name
TMP_DPATH: str = '.keras/datasets'

#: dict[str, Callable]: data loader
LOADER: dict[str, Callable] = {
    'mnist': keras.datasets.mnist.load_data,
    'fashion_mnist': keras.datasets.fashion_mnist.load_data,
    'cifar10': keras.datasets.cifar10.load_data,
    'cifar100': keras.datasets.cifar100.load_data,
}


@dataclass
class ParamDataset:
    """Defines the parameters.
    """
    tmp_dpath: str
    dpath: Path
    loader: Callable
    inputs: np.ndarray
    labels: np.ndarray


def feature_int64_list(value: list[int]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.Int64List``)

    Args:
        value (list[int]): one-dimensional list with elements of type ``int``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [1,2,3]
        print(feature_int64_list(value=value))

        # int64_list {
        # value: 1
        # value: 2
        # value: 3
        # }
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def feature_float_list(value: list[float]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.FloatList``)

    Args:
        value (list[float]): one-dimensional list with elements of type ``float``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [1.,2.,3.]
        print(feature_float_list(value=value))

        # float_list {
        # value: 1
        # value: 2
        # value: 3
        # }
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def feature_bytes_list(value: list[bytes]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.BytesList``)

    Args:
        value (list[bytes]): one-dimensional list with elements of type ``bytes``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [b"1", b"test"]
        print(feature_bytes_list(value=value))

        # bytes_list {
        # value: "1"
        # value: "test"
        # }
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def serialize_example(data: list[list[float]]) -> tf.train.Example:
    """Converts ``tf.train.Example`` type.

    Args:
        data (list[float]): a set of inputs and labels.

    Returns:
        tf.train.Example: value of ``tf.train.Example`` type.
    """
    feature = {
        'input': feature_float_list(data[0]),
        'label': feature_float_list(data[1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_exmaple(args: ParamDataset) -> None:
    """Writes TFRecord data.

    Args:
        args (ParamDataset): parameters.
    """
    fpath = args.dpath
    inputs = np.squeeze(args.inputs)
    labels = np.squeeze(args.labels)

    writer = tf.io.TFRecordWriter(fpath.as_posix())
    for i, (x, y) in enumerate(zip(inputs, labels)):
        example_proto = serialize_example([x.flatten(), [y]])
        writer.write(example_proto.SerializeToString())
        print(f'\rsave progress: {i + 1:>7} / {len(inputs)}', end='')
    print()
    writer.close()


def worker(args: ParamDataset) -> None:
    """Makes TFRecord data.

    *   Make TFRecord data loading data from the following function.

        *   ``keras.datasets.mnist.load_data``
        *   ``keras.datasets.fashion_mnist.load_data``
        *   ``keras.datasets.cifar10.load_data``
        *   ``keras.datasets.cifar100.load_data``

    Args:
        args (ParamDataset): parameters.

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
    dpath = args.dpath
    loader = args.loader

    (x_train, y_train), (x_test, y_test) = loader()

    # training data
    args.dpath = Path(dpath, 'train', 'train.tfr')
    args.dpath.parent.mkdir(parents=True, exist_ok=True)
    args.inputs = x_train
    args.labels = y_train
    write_exmaple(args=args)

    # test data
    args.dpath = Path(dpath, 'test', 'test.tfr')
    args.dpath.parent.mkdir(parents=True, exist_ok=True)
    args.inputs = x_test
    args.labels = y_test
    write_exmaple(args=args)


@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> None:
    """Main.

    Args:
        params (dict[str, Any]): parameters.
    """
    if 'all' in params[K.DATA]:
        params[K.DATA] = list(LOADER.keys())

    args = []
    tmp_dpath = Path(os.environ['HOME'], TMP_DPATH)
    for data_kind in params[K.DATA]:
        dpath = Path(params[K.RESULT], data_kind)
        _args = ParamDataset(
            tmp_dpath=tmp_dpath,
            dpath=dpath,
            loader=LOADER[data_kind],
            inputs=None,
            labels=None,
        )
        args.append(_args)

    with ProcessPoolExecutor(max_workers=params[K.MAX_WORKERS]) as executer:
        executer.map(worker, args)

    shutil.rmtree(tmp_dpath)


def set_params() -> dict[str, Any]:
    """Sets the command line arguments.

    Returns:
        dict[str, Any]: parameters.
    """
    # set the command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        f'--{K.HANDLER}',
        default=[True, True], type=bool, nargs=2,
        help=(
            f'The log handler flag to use.\n'
            f'True: set handler, False: not set handler\n'
            f'ex) --{K.HANDLER} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.LEVEL}',
        default=[20, 20], type=int, nargs=2, choices=[10, 20, 30, 40, 50],
        help=(
            f'The log level.\n'
            f'DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50\n'
            f'ex) --{K.LEVEL} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.PARAM}',
        default='param/param.yaml', type=str,
        help=('The parameter file path.'),
    )
    parser.add_argument(
        f'--{K.RESULT}',
        default='result', type=str,
        help=('The directory path to save the results.'),
    )
    parser.add_argument(
        f'--{K.DATA}',
        default='', type=str, nargs='+', choices=['all', LOADER.keys()],
        help=('The type of data to download.'),
    )
    parser.add_argument(
        f'--{K.MAX_WORKERS}', default=8, type=int,
        help=('The number of download workers.'),
    )

    params = vars(parser.parse_args())

    # # set the file parameters.
    # if params.get(K.PARAM):
    #     fpath = Path(params[K.PARAM])
    #     if fpath.is_file():
    #         params.update(load_yaml(fpath=fpath))

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.HANDLER[PARAM_LOG.SH] = params[K.HANDLER][0]
    PARAM_LOG.HANDLER[PARAM_LOG.FH] = params[K.HANDLER][1]
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if params.get(K.RESULT):
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)
