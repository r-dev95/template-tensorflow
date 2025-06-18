"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.loss import setup

sys.path.append('../tests')
from define import Loss

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupLoss:
    """Tests :class:`setup.SetupLoss`.
    """
    kinds = ['mse', 'cce']
    params = {
        K.LOSS: {
            K.KIND: kinds[0],
            kinds[0]: Loss.MSE,
            kinds[1]: Loss.CCE,
        },
    }
    params_raise = {
        K.LOSS: {
            K.KIND: '',
            '': {},
        },
    }

    labels = [
        "<class 'keras.src.losses.losses.MeanSquaredError'>",
        "<class 'keras.src.losses.losses.CategoricalCrossentropy'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupLoss class does not have a method "{params_raise[K.LOSS][K.KIND]}" that sets the loss function.'),
        ('main', ERROR, f'The available loss function are:'),
    ]
    for key in setup.SetupLoss(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            self.params[K.LOSS][K.KIND] = kind
            _class = setup.SetupLoss(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupLoss(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
