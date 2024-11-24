"""This is the module that tests simple.py.
"""

import sys
from logging import ERROR, INFO, getLogger

import numpy as np
import pytest
import tensorflow as tf
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model import simple

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


DENSE = {
    'units': None,
    'activation': None,
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'lora_rank': None,
}


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    params = {
        K.CLASSES: '',
        K.INPUT_SHAPE: '',
    }

    params_raise = {}

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

        all_log = []
        keys = [K.CLASSES, K.INPUT_SHAPE]
        for key in keys:
            all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

        assert caplog.record_tuples == all_log


class TestSimpleModel:
    """Tests :class:`simple.SimpleModel`.
    """
    params = {
        K.LAYER: {
            K.KIND: ['dense'],
            'dense': DENSE,
        },
        K.CLASSES: {
            K.OPT: '',
            K.LOSS: '',
            K.METRICS: [''],
        },
        K.INPUT_SHAPE: [1],
    }
    params[K.LAYER]['dense']['units'] = 1

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
