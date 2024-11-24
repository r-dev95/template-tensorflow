"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.loss import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


MSE = {
    K.LOSS: {
        K.KIND: 'mse',
        'mse': {
            'reduction': 'sum_over_batch_size',
            'name': 'mean_squared_error',
        },
    },
}
CCE = {
    K.LOSS: {
        K.KIND: 'cce',
        'cce': {
            'from_logits': True,
            'label_smoothing': 0,
            'axis': -1,
            'reduction': 'sum_over_batch_size',
            'name': 'categorical_crossentropy',
        },
    },
}


class TestSetupLoss:
    """Tests :class:`setup.SetupLoss`.
    """

    @pytest.mark.parametrize(
            ('params', 'label'), [
                (MSE, "<class 'keras.src.losses.losses.MeanSquaredError'>"),
                (CCE, "<class 'keras.src.losses.losses.CategoricalCrossentropy'>"),
            ],
    )
    def test(self, params, label):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        _class = setup.SetupLoss(params=params).setup()
        print(f'{type(_class)=}')
        assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.LOSS: {
                K.KIND: '',
                '': {},
            },
        }
        with pytest.raises(ValueError):
            setup.SetupLoss(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupLoss class does not have a method "{params[K.LOSS][K.KIND]}" that sets the loss function.'),
            ('main', ERROR, f'The available loss function are:'),
        ]
        func = setup.SetupLoss(params=MSE).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
