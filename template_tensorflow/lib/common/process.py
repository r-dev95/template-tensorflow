"""This is the module that defines the common process.
"""

import csv
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from lib.common.define import ParamFileName, ParamKey, ParamLog

K = ParamKey()
PARAM_FILE_NAME = ParamFileName()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def fix_random_seed(seed: int) -> None:
    """Fixes the random seed to ensure reproducibility of experiment.

    Args:
        seed (int): random seed.
    """
    keras.utils.set_random_seed(seed=seed)
    tf.config.experimental.enable_op_determinism()


def set_weight(params: dict[str, Any], model: Callable) -> Callable:
    """Sets the model weight.

    Args:
        params (dict[str, Any]): parameters.
        model (Callable): model class.

    Returns:
        Callable: weighted model class.
    """
    fpath = Path(params[K.RESULT], PARAM_FILE_NAME.LOSS)
    with fpath.open(mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    val_loss = np.array(data[1:], dtype=float)[:, data[0].index('val_loss')]
    idx_min = np.argmin(val_loss)
    fpath = list(Path(params[K.RESULT]).glob('*.weights.h5'))[idx_min]

    model.load_weights(fpath)
    return model


def recursive_replace(data: Any, fm_val: Any, to_val: Any) -> Any:  # noqa: ANN401
    """Performs a recursive replacement.

    Args:
        data (Any): data before replacement.
        fm_val (Any): value before replacement.
        to_val (Any): value after replacement.

    Returns:
        Any: data after replacement.
    """
    if isinstance(data, dict):
        return {
            key: recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for key, val in data.items()
        }
    if isinstance(data, list):
        return [
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        ]
    if isinstance(data, tuple):
        return tuple(
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        )
    if isinstance(data, set):
        return {
            recursive_replace(
                data=val,
                fm_val=fm_val,
                to_val=to_val,
            ) for val in data
        }
    if data == fm_val:
        return to_val
    return data
