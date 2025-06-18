"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.metrics import setup

sys.path.append('../tests')
from define import Metrics

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestSetupMetrics:
    """Tests :class:`setup.SetupMetrics`.
    """
    kinds = ['mse', 'cce', 'cacc']
    params = {
        K.METRICS: {
            K.KIND: kinds,
            kinds[0]: Metrics.MSE,
            kinds[1]: Metrics.CCE,
            kinds[2]: Metrics.CACC,
        },
    }
    params_raise = {
        K.METRICS: {
            K.KIND: [''],
            '': {},
        },
    }

    labels = [
        "<class 'keras.src.metrics.reduction_metrics.Mean'>",
        "<class 'keras.src.metrics.regression_metrics.MeanSquaredError'>",
        "<class 'keras.src.metrics.probabilistic_metrics.CategoricalCrossentropy'>",
        "<class 'keras.src.metrics.accuracy_metrics.CategoricalAccuracy'>",
    ]
    all_log = [
        ('main', ERROR, f'SetupMetrics class does not have a method "{params_raise[K.METRICS][K.KIND][0]}" that sets the metrics.'),
        ('main', ERROR, f'The available metrics are:'),
    ]
    for key in setup.SetupMetrics(params=params).func:
        all_log.append(('main', ERROR, f'{key=}'))

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        classes = setup.SetupMetrics(params=self.params).setup()
        for _class, label in zip(classes, self.labels):
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        with pytest.raises(ValueError):
            setup.SetupMetrics(params=self.params_raise).setup()

        assert caplog.record_tuples == self.all_log
