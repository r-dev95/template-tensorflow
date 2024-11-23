"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.metrics import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


MSE = {
    'name': 'mean_squared_error',
}
CCE = {
    'name': 'categorical_crossentropy',
    'from_logits': True,
    'label_smoothing': 0,
    'axis': -1,
}
CACC = {
    'name': 'categorical_accuracy',
}


class TestSetupMetrics:
    """Tests :class:`setup.SetupMetrics`.
    """
    params = {
        K.METRICS: {
            K.KIND: ['mse', 'cce', 'cacc'],
            'mse': MSE,
            'cce': CCE,
            'cacc': CACC,
        },
    }
    labels = [
        "<class 'keras.src.metrics.reduction_metrics.Mean'>",
        "<class 'keras.src.metrics.regression_metrics.MeanSquaredError'>",
        "<class 'keras.src.metrics.probabilistic_metrics.CategoricalCrossentropy'>",
        "<class 'keras.src.metrics.accuracy_metrics.CategoricalAccuracy'>",
    ]

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
        params = {
            K.METRICS: {
                K.KIND: [''],
                '': {},
            },
        }
        with pytest.raises(ValueError):
            setup.SetupMetrics(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupMetrics class does not have a method "{params[K.METRICS][K.KIND][0]}" that sets the metrics.'),
            ('main', ERROR, f'The available metrics are:'),
        ]
        func = setup.SetupMetrics(params=self.params).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
