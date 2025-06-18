"""This is the module that tests train.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

import train
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog

sys.path.append('../tests')
from define import CB, DATA_PARENT_DPATH, Layer, Loss, Metrics, Opt, Proc

PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`train.check_params`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.RESULT: '.',
        K.TRAIN: '.',
        K.VALID: '.',
        K.BATCH_TRAIN: 32,
        K.BATCH_VALID: 1000,
        K.SHUFFLE: 1000,
        K.EPOCHS: 10,
        K.DATA: '',
        K.PROCESS: '',
        K.MODEL: '',
        K.LAYER: '',
        K.OPT: '',
        K.LOSS: '',
        K.METRICS: '',
        K.CB: '',
    }
    params_raise = {
        K.EAGER: 1,
        K.SEED: None,
        K.RESULT: 'dummy',
        K.TRAIN: 'dummy',
        K.VALID: 'dummy',
        K.BATCH_TRAIN: 0,
        K.BATCH_VALID: None,
        K.SHUFFLE: None,
        K.EPOCHS: 0,
    }

    all_log = [
        ('main', ERROR  , f'params["{K.EAGER}"] must be boolean.'),
        ('main', WARNING, f'params["{K.SEED}"] must be integer.'),
        ('main', WARNING, f'The random number seed is not fixed.'),
        ('main', ERROR  , f'params["{K.RESULT}"] is None or the directory does not exists.'),
        ('main', ERROR  , f'params["{K.TRAIN}"] is None or the directory does not exists.'),
        ('main', WARNING, f'params["{K.VALID}"] is None or the directory does not exists.'),
        ('main', WARNING, f'Run without validation data.'),
        ('main', ERROR  , f'params["{K.BATCH_TRAIN}"] must be greater than zero.'),
        ('main', WARNING, f'params["{K.BATCH_VALID}"] must be greater than zero or None.'),
        ('main', WARNING, f'params["{K.SHUFFLE}"] is None.'),
        ('main', WARNING, f'The data is not shuffled.'),
        ('main', ERROR  , f'params["{K.EPOCHS}"] must be greater than zero.'),
    ]
    for key in [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.OPT, K.LOSS, K.METRICS, K.CB]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        train.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            train.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestTrain:
    """Tests :class:`train.Trainer`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.RESULT: 'result',
        K.TRAIN: f'{DATA_PARENT_DPATH}/mnist/train',
        K.VALID: f'{DATA_PARENT_DPATH}/mnist/test',
        K.BATCH_TRAIN: 32,
        K.BATCH_VALID: 1000,
        K.SHUFFLE: None,
        K.EPOCHS: 2,
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
        K.OPT: {
            K.KIND: 'adam',
            'adam': Opt.ADAM,
        },
        K.LOSS: {
            K.KIND: 'cce',
            'cce': Loss.CCE,
        },
        K.METRICS: {
            K.KIND: ['mse'],
            'mse': Metrics.MSE,
        },
        K.CB: {
            K.KIND: ['mcp', 'csv'],
            'mcp': CB.MCP,
            'csv': CB.CSV,
        },
    }

    @pytest.fixture(scope='class')
    def proc(self):
        dpath = Path(self.params[K.RESULT])
        dpath.mkdir(parents=True, exist_ok=True)
        yield
        shutil.rmtree(dpath)

    def test(self, proc):
        """Tests that no errors are raised.
        """
        train.main(params=self.params)
