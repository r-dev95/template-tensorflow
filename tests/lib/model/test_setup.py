"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model import setup

sys.path.append('../tests')
from define import Layer

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupModel:
    """Tests :class:`setup.SetupModel`.
    """
    layers = ['flatten', 'dense_1', 'dense_2', 'conv2d', 'maxpool2d', 'relu']
    kinds = [
        # MLP
        ['flatten', 'dense_1', 'relu', 'dense_2'],
        # CNN
        ['conv2d', 'relu', 'conv2d', 'relu', 'maxpool2d', 'flatten', 'dense_1', 'relu', 'dense_2'],
    ]
    params = {
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: kinds[0],
            layers[0]: Layer.FLATTEN,
            layers[1]: Layer.DENSE_1,
            layers[2]: Layer.DENSE_2,
            layers[3]: Layer.CONV2D,
            layers[4]: Layer.MAXPOOL2D,
            layers[5]: Layer.RELU,
        },
        K.CLASSES: {
            K.OPT: None,
            K.LOSS: None,
            K.METRICS: None,
        },
        K.INPUT_SHAPE: [28, 28, 1],
    }
    params_raise = {
        K.MODEL: {
            K.KIND: '',
            '': {},
        },
    }

    labels = [
        "<class 'lib.model.simple.SimpleModel'>",
        "<class 'lib.model.simple.SimpleModel'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupModel class does not have a method "{params_raise[K.MODEL][K.KIND]}" that sets the model.'),
        ('main', ERROR, f'The available model are:'),
    ]
    for key in setup.SetupModel(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            self.params[K.LAYER][K.KIND] = kind
            _class = setup.SetupModel(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupModel(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
