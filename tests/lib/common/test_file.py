"""This is the module that tests file.py.
"""

import shutil
import sys
from logging import getLogger
from pathlib import Path

import pytest

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common import file
from template_tensorflow.lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestJson:
    """Tests :func:`file.dump_json` and :func:`file.load_json`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: 'data/params.yaml',
        K.TRAIN: None,
        K.RESULT: 'result',
        K.CB: {
            K.KIND: ['mcp'],
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

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.json')
        file.dump_json(data=self.params, fpath=fpath)
        data = file.load_json(fpath=fpath)
        assert self.params == data


class TestYaml:
    """Tests :func:`file.dump_yaml` and :func:`file.load_yaml`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: 'data/params.yaml',
        K.TRAIN: None,
        K.RESULT: 'result',
        K.CB: {
            K.KIND: ['mcp'],
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

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.yaml')
        file.dump_yaml(data=self.params, fpath=fpath)
        data = file.load_yaml(fpath=fpath)
        assert self.params == data


class TestToml:
    """Tests :func:`file.dump_toml` and :func:`file.load_toml`.
    """
    params = {
        K.EAGER: False,
        K.SEED: 0,
        K.PARAM: 'data/params.yaml',
        K.TRAIN: None,
        K.RESULT: 'result',
        K.CB: {
            K.KIND: ['mcp'],
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

        *   The data written matches the data read.
        """
        fpath = Path(self.params[K.RESULT], 'log.toml')
        file.dump_toml(data=self.params, fpath=fpath)
        data = file.load_toml(fpath=fpath)
        assert self.params == data
