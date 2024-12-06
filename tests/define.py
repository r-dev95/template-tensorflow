"""This is the module that defines the test configuration.
"""

import sys
from dataclasses import dataclass
from logging import getLogger

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)

DATA_PARENT_DPATH = 'data'
DATA_RESULT_DPATH = f'{DATA_PARENT_DPATH}/result'

# -----------------------------------------------
# processor parameters
# -----------------------------------------------
@dataclass
class Proc:
    CATENCODE = {
        'num_tokens': 10,
        'output_mode': 'one_hot',
        'sparse': False,
    }
    RESCALE = {
        'scale': 10,
        'offset': 0,
    }
# -----------------------------------------------
# model layer parameters
# -----------------------------------------------
@dataclass
class Layer:
    FLATTEN = {
        'data_format': 'channels_last',
    }
    DENSE_0 = {
        'units': 1,
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
    }
    DENSE_1 = {
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
    }
    DENSE_2 = {
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
    }
    CONV2D = {
        'filters': 8,
        'kernel_size': [3, 3],
        'strides': [1, 1],
        'padding': 'valid',
        'data_format': None,
        'dilation_rate': [1, 1],
        'groups': 1,
        'activation': None,
        'use_bias': True,
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': None,
        'bias_regularizer': None,
        'activity_regularizer': None,
        'kernel_constraint': None,
        'bias_constraint': None,
    }
    MAXPOOL2D = {
        'pool_size': [2, 2],
        'strides': None,
        'padding': 'valid',
        'data_format': None,
        'name': None,
    }
    RELU = {
        'max_value': None,
        'negative_slope': 0,
        'threshold': 0,
    }

# -----------------------------------------------
# optimizer method parameters
# -----------------------------------------------
@dataclass
class Opt:
    SGD = {
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
    }
    ADAM = {
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
    }

# -----------------------------------------------
# loss function parameters
# -----------------------------------------------
@dataclass
class Loss:
    MSE = {
        'reduction': 'sum_over_batch_size',
        'name': 'mean_squared_error',
    }
    CCE = {
        'from_logits': True,
        'label_smoothing': 0,
        'axis': -1,
        'reduction': 'sum_over_batch_size',
        'name': 'categorical_crossentropy',
    }

# -----------------------------------------------
# metrics parameters
# -----------------------------------------------
@dataclass
class Metrics:
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

# -----------------------------------------------
# callback parameters
# -----------------------------------------------
@dataclass
class CB:
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
