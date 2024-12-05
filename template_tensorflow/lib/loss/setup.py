"""This is the module that sets up loss function.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import keras

from lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupLoss` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    kind = params[K.LOSS][K.KIND]
    if kind not in func:
        error = True
        LOGGER.error(
            f'SetupLoss class does not have a method "{kind}" that '
            f'sets the loss function.',
        )
    if error:
        LOGGER.error('The available loss function are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupLoss:
    """Sets up loss function.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'mse': self.mse,
            'cce': self.cce,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up loss function.

        Returns:
            Callable: loss function class.
        """
        kind = self.params[K.LOSS][K.KIND]
        self._params = self.params[K.LOSS][kind]
        loss = self.func[kind]()
        return loss

    def mse(self) -> Callable:
        """Sets ``keras.losses.MeanSquaredError``.

        Returns:
            Callable: loss function class.
        """
        loss = keras.losses.MeanSquaredError(
            reduction=self._params['reduction'],
            name=self._params['name'],
            dtype=None,
        )
        return loss

    def cce(self) -> Callable:
        """Sets ``keras.losses.CategoricalCrossentropy``.

        Returns:
            Callable: loss function class.
        """
        loss = keras.losses.CategoricalCrossentropy(
            from_logits=self._params['from_logits'],
            label_smoothing=self._params['label_smoothing'],
            axis=self._params['axis'],
            reduction=self._params['reduction'],
            name=self._params['name'],
        )
        return loss
