"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.callbacks import setup
from template_tensorflow.lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


MCP = {
    'monitor': 'val_loss',
    'verbose': 0,
    'save_best_only': False,
    'save_weights_only': True,
    'mode': 'auto',
    'save_freq': 'epoch',
    'initial_value_threshold': None,
}
CSV = {
    'separator': ',',
    'append': False,
}


class TestSetupCallbacks:
    """Tests :class:`setup.SetupCallbacks`.
    """
    params = {
        K.RESULT: '.',
        K.CB: {
            K.KIND: ['mcp', 'csv'],
            'mcp': MCP,
            'csv': CSV,
        },
    }
    labels = [
        "<class 'keras.src.callbacks.model_checkpoint.ModelCheckpoint'>",
        "<class 'keras.src.callbacks.csv_logger.CSVLogger'>",
    ]

    def test(self):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        classes = setup.SetupCallbacks(params=self.params).setup()
        for _class, label in zip(classes, self.labels):
            print(f'{type(_class)=}')
            assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.CB: {
                K.KIND: [''],
                '': {},
            },
        }
        with pytest.raises(ValueError):
            setup.SetupCallbacks(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupCallbacks class does not have a method "{params[K.CB][K.KIND][0]}" that sets the callbacks.'),
            ('main', ERROR, f'The available callbacks are:'),
        ]
        func = setup.SetupCallbacks(params=self.params).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
