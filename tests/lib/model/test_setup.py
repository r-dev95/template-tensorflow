"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


_classes = {
    K.OPT: '',
    K.LOSS: '',
    K.METRICS: [''],
}
_flatten = {
    'data_format': 'channels_last',
}
_dense_1 = {
    'units': 100,
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
_dense_2 = {
    'units': 10,
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
_conv2d_1 = {
    'filters': 8,
    'kernel_size': [3, 3],
    'strides': [1, 1],
    'padding': 'same',
    'data_format': None,
    'dilation_rate': [1, 1],
    'groups': 1,
    'activation': None,
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
}
_conv2d_2 = {
    'filters': 16,
    'kernel_size': [3, 3],
    'strides': [1, 1],
    'padding': 'same',
    'data_format': None,
    'dilation_rate': [1, 1],
    'groups': 1,
    'activation': None,
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
}
_maxpool2d = {
    'pool_size': [2, 2],
    'strides': None,
    'padding': 'valid',
    'data_format': None,
    'name': None,
}
_relu = {
    'max_value': None,
    'negative_slope': 0,
    'threshold': 0,
}

MLP = {
    K.MODEL: {
        K.KIND: 'simple',
    },
    K.LAYER: {
        K.KIND: ['flatten', 'dense_1', 'relu', 'dense_2'],
        'flatten': _flatten,
        'dense_1': _dense_1,
        'dense_2': _dense_2,
        'relu': _relu,
    },
    K.CLASSES: _classes,
    K.INPUT_SHAPE: [28, 28, 1],
}
CNN = {
    K.MODEL: {
        K.KIND: 'simple',
    },
    K.LAYER: {
        K.KIND: ['conv2d_1', 'relu', 'conv2d_2', 'relu', 'maxpool2d', 'flatten', 'dense_1', 'relu', 'dense_2'],
        'conv2d_1': _conv2d_1,
        'conv2d_2': _conv2d_2,
        'maxpool2d': _maxpool2d,
        'flatten': _flatten,
        'dense_1': _dense_1,
        'dense_2': _dense_2,
        'relu': _relu,
    },
    K.CLASSES: _classes,
    K.INPUT_SHAPE: [28, 28, 1],
}


class TestSetupModel:
    """Tests :class:`setup.SetupModel`.
    """

    @pytest.mark.parametrize(
            ('params', 'label'), [
                (MLP, "<class 'lib.model.simple.SimpleModel'>"),
                (CNN, "<class 'lib.model.simple.SimpleModel'>"),
            ],
    )
    def test(self, params, label):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        _class = setup.SetupModel(params=params).setup()
        print(f'{type(_class)=}')
        assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.MODEL: {
                K.KIND: '',
                '': {},
            },
        }
        with pytest.raises(ValueError):
            setup.SetupModel(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupModel class does not have a method "{params[K.MODEL][K.KIND]}" that sets the model.'),
            ('main', ERROR, f'The available model are:'),
        ]
        func = setup.SetupModel(params=MLP).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
