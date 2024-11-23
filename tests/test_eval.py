"""This is the module that tests eval.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset, eval  # noqa: A004
from template_tensorflow.lib.common.define import ParamKey

K = ParamKey()


class TestCheckParams:
    """Tests :func:`eval.check_params`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: 'data/params.yaml',
        K.EVAL: '.',
        K.RESULT: '.',
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
        K.PARAM: 'params_raise.yaml',
        K.EVAL: 'data_train',
        K.RESULT: 'result',
        K.BATCH: 0,
    }

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

        all_log = [
            ('main', ERROR  , f'params["{K.EAGER}"] must be boolean.'),
            ('main', WARNING, f'params["{K.SEED}"] must be integer.'),
            ('main', WARNING, f'The random number seed is not fixed.'),
            ('main', ERROR  , f'params["{K.PARAM}"] is None or the file does not exists.'),
            ('main', ERROR  , f'params["{K.EVAL}"] is None or the directory does not exists.'),
            ('main', ERROR  , f'params["{K.RESULT}"] is None or the directory does not exists.'),
            ('main', ERROR  , f'params["{K.BATCH}"] must be greater than zero.'),
        ]
        keys = [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.LOSS, K.METRICS]
        for key in keys:
            all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

        assert caplog.record_tuples == all_log


class TestEval:
    """Tests :class:`eval.evaluator`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: 'data/params.yaml',
        K.EVAL: 'data/mnist/test',
        K.RESULT: 'data/result',
        K.BATCH: 1000,
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
    }

    @pytest.fixture(scope='class')
    def proc(self):
        params = {
            K.RESULT: 'data',
            K.DATA: ['mnist'],
            'max_workers': 8,
        }
        dataset.main(params=params)
        yield
        shutil.rmtree(Path(params[K.RESULT], 'mnist'))

    def test(self, proc):
        """Tests that no errors are raised.
        """
        eval.main(params=self.params)

        Path(self.params[K.RESULT], 'log_params_eval.yaml').unlink()
