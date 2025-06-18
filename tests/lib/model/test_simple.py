"""This is the module that tests simple.py.
"""

import sys
from logging import ERROR, INFO, getLogger

import numpy as np
import pytest
import tensorflow as tf
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model import simple

sys.path.append('../tests')
from define import Layer

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    params = {
        K.CLASSES: '',
        K.INPUT_SHAPE: '',
    }
    params_raise = {}

    all_log = []
    for key in [K.CLASSES, K.INPUT_SHAPE]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        simple.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            simple.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestSimpleModel:
    """Tests :class:`simple.SimpleModel`.
    """
    params = {
        K.LAYER: {
            K.KIND: ['dense'],
            'dense': Layer.DENSE_0,
        },
        K.CLASSES: {
            K.OPT: '',
            K.LOSS: '',
            K.METRICS: [''],
        },
        K.INPUT_SHAPE: [1],
    }

    def test(self):
        """Tests that no errors are raised.

        *   The model makes predictions as expected.
        """
        model = simple.SimpleModel(params=self.params)
        weights = model.layers[0].get_weights()
        weights[0] = np.array([[10]], dtype=int)
        model.set_weights(weights)

        inputs = tf.constant([[1], [2], [3]])
        preds = model(inputs, training=False)
        print(f'{preds=}, {weights=}')

        assert (inputs.numpy() * weights[0] == preds.numpy()).all()
