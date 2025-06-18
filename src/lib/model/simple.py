"""This is the module that builds simple model.
"""

from logging import getLogger
from typing import Any, override

import keras
import tensorflow as tf

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model.base import BaseModel
from lib.model.layer import SetupLayer

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:
    """Checks the :class:`SimpleModel` parameters.

    Args:
        params (dict[str, Any]): parameters.
    """
    error = False # error: True
    keys = [K.CLASSES, K.INPUT_SHAPE]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class SimpleModel(BaseModel):
    """Builds the following simple model.

    *   MLP (Multi Layer Perceptron)
    *   CNN (Convolutional Neural Network)

    Args:
        params (dict[str, Any]): parameters.

    .. attention::

        Since the structure of a class-based model is not defined until input is given,
        ``.summary`` cannot be used.
        For the same reason, trained weights cannot be applied,
        so dummy data is input in ``__init__``.
    """
    def __init__(self, params: dict[str, Any]) -> None:
        check_params(params=params)
        super().__init__(classes=params[K.CLASSES])

        self.model_layers = SetupLayer(params=params).setup()

        dummy_data = keras.layers.Input(shape=params[K.INPUT_SHAPE])
        self(x=dummy_data, training=False)

    @override # type: ignore  # noqa: PGH003
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Outputs the model predictions.

        This method is decorated by ``@override``.

        Args:
            x (tf.Tensor): input.

        Returns:
            tf.Tensor: output.
        """
        for layer in self.model_layers:
            x = layer(x)
        return x
