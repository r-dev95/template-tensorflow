"""This is the module that defines the common process.
"""
from logging import getLogger
from pathlib import Path
from typing import Any

import keras
import pandas as pd
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


def set_weight(params: dict[str, Any], model: keras.models.Model) -> keras.models.Model:
    """Sets the model weight.

    Args:
        params (dict[str, Any]): parameters.
        model (keras.models.Model): model class.

    Returns:
        keras.models.Model: weighted model class.
    """
    fpath = Path(params[K.RESULT], PARAM_FILE_NAME.LOSS)
    df = pd.read_csv(fpath)
    idx_min = int(df['val_loss'].argmin())
    idx_min = int(df.loc[idx_min]['epoch'])

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
