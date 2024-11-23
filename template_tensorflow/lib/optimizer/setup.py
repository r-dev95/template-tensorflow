"""This is the module that sets up optimizer method.
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
    """Checks the :class:`SetupOpt` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    kind = params[K.OPT][K.KIND]
    if kind not in func:
        error = True
        LOGGER.error(
            f'SetupOpt class does not have a method "{kind}" that '
            f'sets the optimizer method.',
        )
    if error:
        LOGGER.error('The available optimizer method are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupOpt:
    """Sets up optimizer method.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__()`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'sgd': self.sgd,
            'adam': self.adam,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up optimizer method.

        Returns:
            Callable: optimizer method class.
        """
        kind = self.params[K.OPT][K.KIND]
        self._params = self.params[K.OPT][kind]
        opt = self.func[kind]()
        return opt

    def sgd(self) -> Callable:
        """Sets ``keras.optimizers.SGD``.

        Returns:
            Callable: optimizer method class.
        """
        opt = keras.optimizers.SGD(
            learning_rate=self._params['learning_rate'],
            momentum=self._params['momentum'],
            nesterov=self._params['nesterov'],
            weight_decay=self._params['weight_decay'],
            clipnorm=self._params['clipnorm'],
            clipvalue=self._params['clipvalue'],
            global_clipnorm=self._params['global_clipnorm'],
            use_ema=self._params['use_ema'],
            ema_momentum=self._params['ema_momentum'],
            ema_overwrite_frequency=self._params['ema_overwrite_frequency'],
            loss_scale_factor=self._params['loss_scale_factor'],
            gradient_accumulation_steps=self._params['gradient_accumulation_steps'],
            name=self._params['name'],
        )
        return opt

    def adam(self) -> Callable:
        """Sets ``keras.optimizers.Adam``.

        Returns:
            Callable: optimizer method class.
        """
        opt = keras.optimizers.Adam(
            learning_rate=self._params['learning_rate'],
            beta_1=self._params['beta_1'],
            beta_2=self._params['beta_2'],
            epsilon=self._params['epsilon'],
            amsgrad=self._params['amsgrad'],
            weight_decay=self._params['weight_decay'],
            clipnorm=self._params['clipnorm'],
            clipvalue=self._params['clipvalue'],
            global_clipnorm=self._params['global_clipnorm'],
            use_ema=self._params['use_ema'],
            ema_momentum=self._params['ema_momentum'],
            ema_overwrite_frequency=self._params['ema_overwrite_frequency'],
            loss_scale_factor=self._params['loss_scale_factor'],
            gradient_accumulation_steps=self._params['gradient_accumulation_steps'],
            name=self._params['name'],
        )
        return opt
