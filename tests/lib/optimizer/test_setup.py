"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.optimizer import setup

sys.path.append('../tests')
from define import Opt

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupOpt:
    """Tests :class:`setup.SetupOpt`.
    """
    kinds = ['sgd', 'adam']
    params = {
        K.OPT: {
            K.KIND: kinds[0],
            kinds[0]: Opt.SGD,
            kinds[1]: Opt.ADAM,
        },
    }
    params_raise = {
        K.OPT: {
            K.KIND: '',
            '': {},
        },
    }

    labels = [
        "<class 'keras.src.optimizers.sgd.SGD'>",
        "<class 'keras.src.optimizers.adam.Adam'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupOpt class does not have a method "{params_raise[K.OPT][K.KIND]}" that sets the optimizer method.'),
        ('main', ERROR, f'The available optimizer method are:'),
    ]
    for key in setup.SetupOpt(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        for kind, label in zip(self.kinds, self.labels):
            self.params[K.OPT][K.KIND] = kind
            _class = setup.SetupOpt(params=self.params).setup()
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupOpt(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
