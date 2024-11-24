"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.optimizer import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


SGD = {
    K.OPT: {
        K.KIND: 'sgd',
        'sgd': {
            'learning_rate': 0.001,
            'momentum': 0.0,
            'nesterov': False,
            'weight_decay': None,
            'clipnorm': None,
            'clipvalue': None,
            'global_clipnorm': None,
            'use_ema': False,
            'ema_momentum': 0.99,
            'ema_overwrite_frequency': None,
            'loss_scale_factor': None,
            'gradient_accumulation_steps': None,
            'name': 'SGD',
        },
    },
}
ADAM = {
    K.OPT: {
        K.KIND: 'adam',
        'adam': {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 0.0000001,
            'amsgrad': False,
            'weight_decay': None,
            'clipnorm': None,
            'clipvalue': None,
            'global_clipnorm': None,
            'use_ema': False,
            'ema_momentum': 0.99,
            'ema_overwrite_frequency': None,
            'loss_scale_factor': None,
            'gradient_accumulation_steps': None,
            'name': 'adam',
        },
    },
}


class TestSetupOpt:
    """Tests :class:`setup.SetupOpt`.
    """

    @pytest.mark.parametrize(
            ('params', 'label'), [
                (SGD , "<class 'keras.src.optimizers.sgd.SGD'>"),
                (ADAM, "<class 'keras.src.optimizers.adam.Adam'>"),
            ],
    )
    def test(self, params, label):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        _class = setup.SetupOpt(params=params).setup()
        print(f'{type(_class)=}')
        assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.OPT: {
                K.KIND: '',
                '': {},
            },
        }
        with pytest.raises(ValueError):
            setup.SetupOpt(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupOpt class does not have a method "{params[K.OPT][K.KIND]}" that sets the optimizer method.'),
            ('main', ERROR, f'The available optimizer method are:'),
        ]
        func = setup.SetupOpt(params=SGD).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
