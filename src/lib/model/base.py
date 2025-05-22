"""This is the module that defines the base model.
"""  # noqa: INP001

from collections.abc import Callable
from logging import getLogger
from typing import override

import keras
import tensorflow as tf

from lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(classes: dict[str, Callable]) -> None:
    """Checks the :class:`BaseModel` parameters.

    Args:
        classes (dict[str, Callable]): class list.
    """
    error = False # error: True
    keys = [K.OPT, K.LOSS, K.METRICS]
    for key in keys:
        if key not in classes:
            error = True
            LOGGER.error(f'The key "{key}" for variable "classes" is missing.')
    if error:
        raise ValueError


class BaseModel(keras.models.Model):
    """Defines the base model.

    *   You can customize :meth:`train_step` and :meth:`test_step` using ``.fit``.
    *   In eager mode, you can output calculation results using ``print`` or logging
        in :meth:`train_step`, :meth:`test_step`, or ``.call`` of class-form models.
    *   In graph mode, you can output too.
        But you will need to implement a custom layer that ``tf.print`` the input.

    Args:
        classes (dict[str, Callable]): class list.
    """

    def __init__(self, classes: dict[str, Callable]) -> None:  # noqa: ANN401
        self.classes = classes
        check_params(classes=classes)
        super().__init__()

    def update_metrics(self, data: tuple[tf.Tensor]) -> dict[str, float]:
        """Updates metrics.

        Args:
            data (tuple[tf.Tensor]): tuple of labels, preds, losses, and sample_weight.

        Returns:
            dict[str, float]: all metrics results.
        """
        labels, preds, losses, sample_weight = data
        for m in self.classes[K.METRICS]:
            if not isinstance(m, keras.metrics.MeanMetricWrapper):
                m(losses, sample_weight=sample_weight)
            else:
                m(labels, preds, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.classes[K.METRICS]}

    @override # type: ignore  # noqa: PGH003
    @property
    def metrics(self) -> list[Callable]:
        """Returns list of metrics classes.

        This function is decorated by ``@override`` and ``@property``.

        *   When using ``.fit`` or ``.evaluate``, Metrics must be run
            ``.reset_state`` at the start of an epoch.
            By setting the return value of this method to a list of all metrics, it will
            automatically run ``.reset_state``.

        Returns:
            list[Callable]: list of metrics classes.
        """
        return self.classes[K.METRICS]

    @override # type: ignore  # noqa: PGH003
    def train_step(self, data: tuple[tf.Tensor]) -> dict[str, float]:
        """Trains the model one step at a time.

        This function is decorated by ``@override``.

        #.  Output predictions. (forward propagation)
        #.  Output losses.
        #.  Output gradients and update model parameters. (back propagation)
        #.  Update metrics.

        Args:
            data (tuple[tf.Tensor]):
                tuple of inputs and labels (and weights for each input).

        Returns:
            dict[str, float]: all metrics results.
        """
        inputs, labels, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            preds = self(x=inputs, training=True)
            losses = self.classes[K.LOSS](
                y_true=labels,
                y_pred=preds,
                sample_weight=sample_weight,
            )
        grads = tape.gradient(
            target=losses,
            sources=self.trainable_variables,
        )
        self.classes[K.OPT].apply_gradients(
            zip(grads, self.trainable_variables),
        )
        res = self.update_metrics(data=(labels, preds, losses, sample_weight))
        return res

    @override # type: ignore  # noqa: PGH003
    def test_step(self, data: tuple[tf.Tensor]) -> dict[str, float]:
        """Validations the model one step at a time.

        This function is decorated by ``@override``.

        #.  Output predictions. (forward propagation)
        #.  Output losses.
        #.  Update metrics.

        Args:
            data (tuple[tf.Tensor]):
                tuple of inputs and labels (and weights for each input).

        Returns:
            dict[str, float]: all metrics results.
        """
        inputs, labels, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x=inputs, training=False)
        losses = self.classes[K.LOSS](
            y_true=labels,
            y_pred=preds,
            sample_weight=sample_weight,
        )
        res = self.update_metrics(data=(labels, preds, losses, sample_weight))
        return res
