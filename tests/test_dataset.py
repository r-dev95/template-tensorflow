"""This is the module that tests eval.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING
from pathlib import Path

import pytest

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset
from template_tensorflow.lib.common.define import ParamKey

K = ParamKey()


class TestDataset:
    """Tests :mod:`dataset`.
    """
    params = {
        K.RESULT: 'result',
        K.DATA: ['all'],
        'max_workers': 8,
    }

    @pytest.fixture(scope='class')
    def proc(self):
        yield
        shutil.rmtree(self.params[K.RESULT])

    def test(self, proc):
        """Tests that no errors are raised.
        """
        dataset.main(params=self.params)

        # mnist
        fpath = Path(self.params[K.RESULT], 'mnist', 'train', 'train.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
        fpath = Path(self.params[K.RESULT], 'mnist', 'test', 'test.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0

        # fashion_mnist
        fpath = Path(self.params[K.RESULT], 'fashion_mnist', 'train', 'train.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
        fpath = Path(self.params[K.RESULT], 'fashion_mnist', 'test', 'test.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0

        # cifar10
        fpath = Path(self.params[K.RESULT], 'cifar10', 'train', 'train.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
        fpath = Path(self.params[K.RESULT], 'cifar10', 'test', 'test.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0

        # cifar100
        fpath = Path(self.params[K.RESULT], 'cifar100', 'train', 'train.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
        fpath = Path(self.params[K.RESULT], 'cifar100', 'test', 'test.tfr')
        assert fpath.is_file()
        assert fpath.stat().st_size > 0
