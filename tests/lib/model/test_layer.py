"""This is the module that tests layer.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.model import layer

sys.path.append('../tests')
from define import Layer

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupLayer:
    """Tests :class:`layer.SetupLayer`.
    """
    kinds = ['flatten', 'dense', 'conv2d', 'maxpool2d', 'relu']
    params = {
        K.LAYER: {
            K.KIND: kinds,
            kinds[0]: Layer.FLATTEN,
            kinds[1]: Layer.DENSE_1,
            kinds[2]: Layer.CONV2D,
            kinds[3]: Layer.MAXPOOL2D,
            kinds[4]: Layer.RELU,
        },
    }
    params_raise = {
        K.LAYER: {
            K.KIND: [''],
            '': {},
        },
    }

    labels = [
        "<class 'keras.src.layers.reshaping.flatten.Flatten'>",
        "<class 'keras.src.layers.core.dense.Dense'>",
        "<class 'keras.src.layers.convolutional.conv2d.Conv2D'>",
        "<class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>",
        "<class 'keras.src.layers.activations.relu.ReLU'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupLayer class does not have a method "{params_raise[K.LAYER][K.KIND][0]}" that sets the model layer.'),
        ('main', ERROR, f'The available model layer are:'),
    ]
    for key in layer.SetupLayer(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

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
        with pytest.raises(ValueError):
            layer.SetupLayer(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
