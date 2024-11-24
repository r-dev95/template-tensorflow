"""This is the module that tests train.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow import train
from template_tensorflow.lib.common.define import ParamKey, ParamLog

sys.path.append('../tests')
from define import DATA_PARAMS_FPATH, DATA_PARENT_DPATH

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`train.check_params`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: DATA_PARAMS_FPATH,
        K.TRAIN: '.',
        K.VALID: '.',
        K.RESULT: '.',
        K.EPOCHS: 10,
        K.BATCH_TRAIN: 32,
        K.BATCH_VALID: 1000,
        K.SHUFFLE: 1000,
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
        K.PARAM: 'dummy.yaml',
        K.TRAIN: 'dummy',
        K.VALID: 'dummy',
        K.RESULT: 'dummy',
        K.EPOCHS: 0,
        K.BATCH_TRAIN: 0,
        K.BATCH_VALID: None,
        K.SHUFFLE: None,
    }

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

        all_log = [
            ('main', ERROR  , f'params["{K.EAGER}"] must be boolean.'),
            ('main', WARNING, f'params["{K.SEED}"] must be integer.'),
            ('main', WARNING, f'The random number seed is not fixed.'),
            ('main', ERROR  , f'params["{K.PARAM}"] is None or the file does not exists.'),
            ('main', ERROR  , f'params["{K.TRAIN}"] is None or the directory does not exists.'),
            ('main', WARNING, f'params["{K.VALID}"] is None or the directory does not exists.'),
            ('main', WARNING, f'Run without validation data.'),
            ('main', ERROR  , f'params["{K.RESULT}"] is None or the directory does not exists.'),
            ('main', ERROR  , f'params["{K.EPOCHS}"] must be greater than zero.'),
            ('main', ERROR  , f'params["{K.BATCH_TRAIN}"] must be greater than zero.'),
            ('main', WARNING, f'params["{K.BATCH_VALID}"] must be greater than zero or None.'),
            ('main', WARNING, f'params["{K.SHUFFLE}"] is None.'),
            ('main', WARNING, f'The data is not shuffled.'),
        ]
        keys = [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.OPT, K.LOSS, K.METRICS, K.CB]
        for key in keys:
            all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

        assert caplog.record_tuples == all_log


class TestTrain:
    """Tests :class:`train.Trainer`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: DATA_PARAMS_FPATH,
        K.TRAIN: f'{DATA_PARENT_DPATH}/mnist/train',
        K.VALID: f'{DATA_PARENT_DPATH}/mnist/test',
        K.RESULT: 'result',
        K.EPOCHS: 2,
        K.BATCH_TRAIN: 32,
        K.BATCH_VALID: 1000,
        K.SHUFFLE: None,
        K.DATA: {K.KIND: 'mnist'},
        K.PROCESS: {
            K.KIND: ['catencode', 'rescale'],
            'catencode': {
                'num_tokens': 10,
                'output_mode': 'one_hot',
                'sparse': False,
            },
            'rescale': {
                'scale': 0.00392156862745098,
                'offset': 0,
            },
        },
        K.MODEL: {K.KIND: 'simple'},
        K.LAYER: {
            K.KIND: ['flatten', 'dense_1', 'relu', 'dense_2'],
            'flatten': {'data_format': 'channels_last'},
            'dense_1': {
                'units': 100,
                'activation': None,
                'use_bias': True,
                'kernel_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros',
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None,
                'lora_rank': None,
            },
            'dense_2': {
                'units': 10,
                'activation': None,
                'use_bias': True,
                'kernel_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros',
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None,
                'lora_rank': None,
            },
            'relu': {
                'max_value': None,
                'negative_slope': 0,
                'threshold': 0,
            },
        },
        K.OPT: {
            K.KIND: 'adam',
            'adam': {
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-07,
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
        K.METRICS: {
            K.KIND: ['mse'],
            'mse': {'name': 'mean_squared_error'},
        },
        K.CB: {
            K.KIND: ['mcp', 'csv'],
            'mcp': {
                # 'filepath': 'epoch{epoch:03d}_loss{loss:.3f}_{val_loss:.3f}.weights.h5',
                'monitor': 'val_loss',
                'verbose': 0,
                'save_best_only': False,
                'save_weights_only': True,
                'mode': 'auto',
                'save_freq': 'epoch',
                'initial_value_threshold': None,
            },
            'csv': {
                # 'filename': 'log_loss.csv',
                'separator': ',',
                'append': False,
            },
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
