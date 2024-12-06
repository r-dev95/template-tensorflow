"""This is the module that tests eval.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow import eval  # noqa: A004
from template_tensorflow.lib.common.define import ParamKey, ParamLog

sys.path.append('../tests')
from define import DATA_RESULT_DPATH, DATA_PARENT_DPATH, Layer, Loss, Metrics, Proc

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`eval.check_params`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.RESULT: '.',
        K.EVAL: '.',
        K.BATCH: 1000,
        K.DATA: '',
        K.PROCESS: '',
        K.MODEL: '',
        K.LAYER: '',
        K.LOSS: '',
        K.METRICS: '',
        K.CB: '',
    }
    params_raise = {
        K.EAGER: 1,
        K.SEED: None,
        K.RESULT: 'dummy',
        K.EVAL: 'dummy',
        K.BATCH: 0,
    }

    all_log = [
        ('main', ERROR  , f'params["{K.EAGER}"] must be boolean.'),
        ('main', WARNING, f'params["{K.SEED}"] must be integer.'),
        ('main', WARNING, f'The random number seed is not fixed.'),
        ('main', ERROR  , f'params["{K.RESULT}"] is None or the directory does not exists.'),
        ('main', ERROR  , f'params["{K.EVAL}"] is None or the directory does not exists.'),
        ('main', ERROR  , f'params["{K.BATCH}"] must be greater than zero.'),
    ]
    for key in [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.LOSS, K.METRICS]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        eval.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            eval.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestEval:
    """Tests :class:`eval.evaluator`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.RESULT: DATA_RESULT_DPATH,
        K.EVAL: f'{DATA_PARENT_DPATH}/mnist/test',
        K.BATCH: 1000,
        K.DATA: {K.KIND: 'mnist'},
        K.PROCESS: {
            K.KIND: ['catencode', 'rescale'],
            'catencode': Proc.CATENCODE,
            'rescale': Proc.RESCALE,
        },
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: ['flatten', 'dense_1', 'relu', 'dense_2'],
            'flatten': Layer.FLATTEN,
            'dense_1': Layer.DENSE_1,
            'dense_2': Layer.DENSE_2,
            'relu': Layer.RELU,
        },
        K.LOSS: {
            K.KIND: 'cce',
            'cce': Loss.CCE,
        },
        K.METRICS: {
            K.KIND: ['mse'],
            'mse': Metrics.MSE,
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        yield
        Path(self.params[K.RESULT], 'log_params_eval.yaml').unlink()

    def test(self, proc):
        """Tests that no errors are raised.
        """
        eval.main(params=self.params)
