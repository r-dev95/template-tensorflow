"""This is the module that sets up metrics.
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
    """Checks the :class:`SetupMetrics` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.METRICS][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
                f'SetupMetrics class does not have a method "{kind}" that '
                f'sets the metrics.',
            )
    if error:
        LOGGER.error('The available metrics are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupMetrics:
    """Sets up metrics.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__()`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'mse': self.mse,
            'cce': self.cce,
            'cacc': self.cacc,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up metrics.

        Returns:
            Callable: metrics class.
        """
        metrics = [keras.metrics.Mean(name='loss')]
        for kind in self.params[K.METRICS][K.KIND]:
            self._params = self.params[K.METRICS][kind]
            metrics.append(self.func[kind]())
        return metrics

    def mse(self) -> Callable:
        """Sets ``keras.metrics.MeanSquaredError``.

        Returns:
            Callable: metrics class.
        """
        metrics = keras.metrics.MeanSquaredError(
            name=self._params['name'],
            dtype=None,
        )
        return metrics

    def cce(self) -> Callable:
        """Sets ``keras.metrics.CategoricalCrossentropy``.

        Returns:
            Callable: metrics class.
        """
        metrics = keras.metrics.CategoricalCrossentropy(
            name=self._params['name'],
            dtype=None,
            from_logits=self._params['from_logits'],
            label_smoothing=self._params['label_smoothing'],
            axis=self._params['axis'],
        )
        return metrics

    def cacc(self) -> Callable:
        """Sets ``keras.metrics.CategoricalAccuracy``.

        Returns:
            Callable: metrics class.
        """
        metrics = keras.metrics.CategoricalAccuracy(
            name=self._params['name'],
            dtype=None,
        )
        return metrics
