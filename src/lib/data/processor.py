"""This is the module that process data.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import keras
import tensorflow as tf

from lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`Processor` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.PROCESS][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
            f'Processor class does not have a method "{kind}" that '
            f'sets the processing method.',
        )
    if error:
        LOGGER.error('The available processing method are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class Processor:
    """Processes data.

    *   Used to process data when making a ``tf.data`` data pipeline.
    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'catencode': self.catencode,
            'rescale': self.rescale,
        }
        check_params(params=params, func=self.func)

    def run(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs process data.

        Args:
            x (tf.Tensor): input. (before process)
            y (tf.Tensor): label. (before process)

        Returns:
            tf.Tensor: input. (after process)
            tf.Tensor: label. (after process)
        """
        for kind in self.params[K.PROCESS][K.KIND]:
            self._params = self.params[K.PROCESS][kind]
            x, y = self.func[kind](x, y)
        return x, y

    def catencode(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs ``keras.layers.CategoryEncoding``.

        Args:
            x (tf.Tensor): input. (before process)
            y (tf.Tensor): label. (before process)

        Returns:
            tf.Tensor: input. (after process)
            tf.Tensor: label. (after process)
        """
        y = keras.layers.CategoryEncoding(
            num_tokens=self._params['num_tokens'],
            output_mode=self._params['output_mode'],
            sparse=self._params['sparse'],
        )(y)
        y = tf.squeeze(y)
        return x, y

    def rescale(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs ``keras.layers.Rescaling``.

        Args:
            x (tf.Tensor): input. (before process)
            y (tf.Tensor): label. (before process)

        Returns:
            tf.Tensor: input. (after process)
            tf.Tensor: label. (after process)
        """
        x = keras.layers.Rescaling(
            scale=self._params['scale'],
            offset=self._params['offset'],
        )(x)
        return x, y
