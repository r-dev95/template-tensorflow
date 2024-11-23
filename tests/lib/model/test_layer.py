"""This is the module that tests layer.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model import layer

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


FLATTEN = {
    'data_format': 'channels_last',
}
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
CONV2D = {
    'filters': 8,
    'kernel_size': [3, 3],
    'strides': [1, 1],
    'padding': 'valid',
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
MAXPOOL2D = {
    'pool_size': [2, 2],
    'strides': None,
    'padding': 'valid',
    'data_format': None,
    'name': None,
}
RELU = {
    'max_value': None,
    'negative_slope': 0,
    'threshold': 0,
}


class TestSetupLayer:
    """Tests :class:`layer.SetupLayer`.
    """
    params = {
        K.LAYER: {
            K.KIND: ['flatten', 'dense', 'conv2d', 'maxpool2d', 'relu'],
            'flatten': FLATTEN,
            'dense': DENSE,
            'conv2d': CONV2D,
            'maxpool2d': MAXPOOL2D,
            'relu': RELU,
        },
    }
    labels = [
        "<class 'keras.src.layers.reshaping.flatten.Flatten'>",
        "<class 'keras.src.layers.core.dense.Dense'>",
        "<class 'keras.src.layers.convolutional.conv2d.Conv2D'>",
        "<class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>",
        "<class 'keras.src.layers.activations.relu.ReLU'>",
    ]

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        classes = layer.SetupLayer(params=self.params).setup()
        for _class, label in zip(classes, self.labels):
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.LAYER: {
                K.KIND: [''],
                '': {},
            },
        }
        with pytest.raises(ValueError):
            layer.SetupLayer(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupLayer class does not have a method "{params[K.LAYER][K.KIND][0]}" that sets the model layer.'),
            ('main', ERROR, f'The available model layer are:'),
        ]
        func = layer.SetupLayer(params=self.params).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
