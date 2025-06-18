"""This is the module that sets up model layers.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any

import keras

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any], func: dict[str, Callable]) -> None:
    """Checks the :class:`SetupLayer` parameters.

    Args:
        params (dict[str, Any]): parameters.
        func (dict[str, Callable]): Class variables whose values are available methods.
    """
    error = False # error: True
    for kind in params[K.LAYER][K.KIND]:
        layer = kind.split('_')[0]
        if layer not in func:
            error = True
            LOGGER.error(
                f'SetupLayer class does not have a method "{kind}" that '
                f'sets the model layer.',
            )
    if error:
        LOGGER.error('The available model layer are:')
        for key in func:
            LOGGER.error(f'{key=}')
        raise ValueError


class SetupLayer:
    """Sets up the model layer.

    *   If you want to use some other settings, implement it as a method of this class.
        If you implemented, set the name as the ``func`` key in ``__init__`` and the
        method as the value.

    Args:
        params (dict[str, Any]): parameters.
    """
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.func = {
            'flatten': self.flatten,
            'dense': self.dense,
            'conv2d': self.conv2d,
            'maxpool2d': self.maxpool2d,
            'relu': self.relu,
        }
        check_params(params=params, func=self.func)

    def setup(self) -> list[Callable]:
        """Sets up model layers.

        Returns:
            list[Callable]: list of model layers.
        """
        layers = []
        for layer in self.params[K.LAYER][K.KIND]:
            _layer = layer.split('_')[0]
            self._params = self.params[K.LAYER][layer]
            layers.append(
                self.func[_layer](),
            )
        return layers

    def flatten(self) -> Callable:
        """Sets ``keras.layers.Flatten``.

        Returns:
            Callable: model layer class.
        """
        layer = keras.layers.Flatten(data_format=self._params['data_format'])
        return layer

    def dense(self) -> Callable:
        """Sets ``keras.layers.Dense``.

        Returns:
            Callable: model layer class.
        """
        layer = keras.layers.Dense(
            units=self._params['units'],
            activation=None,
            use_bias=self._params['use_bias'],
            kernel_initializer=self._params['kernel_initializer'],
            bias_initializer=self._params['bias_initializer'],
            kernel_regularizer=self._params['kernel_regularizer'],
            bias_regularizer=self._params['bias_regularizer'],
            activity_regularizer=self._params['activity_regularizer'],
            kernel_constraint=self._params['kernel_constraint'],
            bias_constraint=self._params['bias_constraint'],
            lora_rank=self._params['lora_rank'],
        )
        return layer

    def conv2d(self) -> Callable:
        """Sets ``keras.layers.Conv2D``.

        Returns:
            Callable: model layer class.
        """
        layer = keras.layers.Conv2D(
            filters=self._params['filters'],
            kernel_size=self._params['kernel_size'],
            strides=self._params['strides'],
            padding=self._params['padding'],
            data_format=self._params['data_format'],
            dilation_rate=self._params['dilation_rate'],
            groups=self._params['groups'],
            activation=self._params['activation'],
            use_bias=self._params['use_bias'],
            kernel_initializer=self._params['kernel_initializer'],
            bias_initializer=self._params['bias_initializer'],
            kernel_regularizer=self._params['kernel_regularizer'],
            bias_regularizer=self._params['bias_regularizer'],
            activity_regularizer=self._params['activity_regularizer'],
            kernel_constraint=self._params['kernel_constraint'],
            bias_constraint=self._params['bias_constraint'],
        )
        return layer

    def maxpool2d(self) -> Callable:
        """Sets ``keras.layers.MaxPool2D``.

        Returns:
            Callable: model layer class.
        """
        layer = keras.layers.MaxPool2D(
            pool_size=self._params['pool_size'],
            strides=self._params['strides'],
            padding=self._params['padding'],
            data_format=self._params['data_format'],
            name=self._params['data_format'],
        )
        return layer

    def relu(self) -> Callable:
        """Sets ``keras.layers.ReLU``.

        Returns:
            Callable: model layer class.
        """
        layer = keras.layers.ReLU(
            max_value=self._params['max_value'],
            negative_slope=self._params['negative_slope'],
            threshold=self._params['threshold'],
        )
        return layer
