"""This is the module that sets up callbacks.
"""

from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any

import keras

from lib.common.define import ParamFileName, ParamKey, ParamLog

K = ParamKey()
PARAM_FILE_NAME = ParamFileName()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupCallbacks` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.CB][K.KIND]:
        if kind not in func:
            error = True
            LOGGER.error(
                f'SetupCallbacks class does not have a method "{kind}" that '
                f'sets the callbacks.',
            )
    if error:
        LOGGER.error('The available callbacks are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupCallbacks:
    """Sets up callbacks.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__()`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'mcp': self.mcp,
            'csv': self.csv,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> Callable:
        """Sets up callbacks.

        Returns:
            Callable: callbacks class.
        """
        callbacks = []
        for kind in self.params[K.CB][K.KIND]:
            self._params = self.params[K.CB][kind]
            callbacks.append(self.func[kind]())
        return callbacks

    def mcp(self) -> Callable:
        """Sets ``keras.callbacks.ModelCheckpoint``.

        Returns:
            Callable: callbacks class.
        """
        # fpath = Path(self.params[K.RESULT], self._params['filepath'])
        fpath = Path(self.params[K.RESULT], PARAM_FILE_NAME.WIGHT)
        callbacks = keras.callbacks.ModelCheckpoint(
            filepath=fpath,
            monitor=self._params['monitor'],
            verbose=self._params['verbose'],
            save_best_only=self._params['save_best_only'],
            save_weights_only=self._params['save_weights_only'],
            mode=self._params['mode'],
            save_freq=self._params['save_freq'],
            initial_value_threshold=self._params['initial_value_threshold'],
        )
        return callbacks

    def csv(self) -> Callable:
        """Sets ``keras.callbacks.CSVLogger``.

        Returns:
            Callable: callbacks class.
        """
        # fpath = Path(self.params[K.RESULT], self._params['filename'])
        fpath = Path(self.params[K.RESULT], PARAM_FILE_NAME.LOSS)
        callbacks = keras.callbacks.CSVLogger(
            filename=fpath,
            separator=self._params['separator'],
            append=self._params['append'],
        )
        return callbacks
