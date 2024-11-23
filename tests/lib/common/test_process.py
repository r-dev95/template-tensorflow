"""This is the module that tests process.py.
"""

import random
import sys
from logging import getLogger

import numpy as np
import tensorflow as tf

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common import process
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model.setup import SetupModel

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestFixRandomSeed:
    """Tests :func:`process.fix_random_seed`.
    """

    def test(self):
        """Tests that no errors are raised.

        *   The random number seed is fixed.
        """
        # random
        process.fix_random_seed(seed=0)
        before = random.randint(0, 10)
        process.fix_random_seed(seed=0)
        after = random.randint(0, 10)
        print(f'{before=}, {after=}')
        assert before == after

        # np.random
        process.fix_random_seed(seed=0)
        before = np.random.randint(0, 10)
        process.fix_random_seed(seed=0)
        after = np.random.randint(0, 10)
        print(f'{before=}, {after=}')
        assert before == after

        # tf.random
        process.fix_random_seed(seed=0)
        before = tf.random.uniform(shape=[1], minval=0, maxval=10)
        process.fix_random_seed(seed=0)
        after = tf.random.uniform(shape=[1], minval=0, maxval=10)
        print(f'{before=}, {after=}')
        assert before == after


class TestSetWeight:
    """Tests :func:`process.set_weight`.
    """
    params = {
        K.EAGER: False,
        K.RESULT: 'data/result',
        K.MODEL: {
            K.KIND: 'simple',
        },
        K.LAYER: {
            K.KIND: ['flatten', 'dense_1', 'relu', 'dense_2'],
            'flatten': {
                'data_format': 'channels_last',
            },
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
    }
    classes = {
        K.OPT: '',
        K.LOSS: '',
        K.METRICS: [''],
    }

    def test(self):
        """Tests that no errors are raised.

        *   The initialization weights are replaced by the trained weights.
        """
        self.params[K.CLASSES] = self.classes
        self.params[K.INPUT_SHAPE] = [28, 28, 1]

        process.fix_random_seed(seed=0)
        model = SetupModel(params=self.params).setup()

        process.fix_random_seed(seed=0)
        weighted_model = SetupModel(params=self.params).setup()
        weighted_model = process.set_weight(params=self.params, model=weighted_model)

        for m, w_m in zip(model.layers, weighted_model.layers):
            if m.get_weights():
                for m_weight, w_m_weight in zip(m.get_weights(), w_m.get_weights()):
                    # print(np.where(m_weight != w_m_weight))
                    # print(f'{m_weight=}')
                    # print(f'{w_m_weight=}')
                    assert (m_weight != w_m_weight).any()


class TestRecursiveReplace:
    """Tests :func:`process.recursive_replace`.
    """
    params = {
        'aaa': None,
        'bbb': {
            'ccc': (None, None),
        },
        'ddd': {
            'eee': [None],
            'fff': {
                'ggg': None,
            },
        },
    }

    def test(self):
        """Tests that no errors are raised.

        *   The replaced value must match the inverse replaced value.
        """
        data = process.recursive_replace(data=self.params, fm_val=None, to_val='None')
        print(f'{data=}')
        data = process.recursive_replace(data=self.params, fm_val='None', to_val=None)
        assert self.params == data
