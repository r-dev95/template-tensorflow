"""This is the module that tests eval.py.
"""

import sys
from logging import getLogger
from pathlib import Path

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset
from template_tensorflow.lib.common.define import ParamKey, ParamLog

sys.path.append('../tests')
from define import DATA_PARENT_DPATH

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestDataset:
    """Tests :mod:`dataset`.
    """
    params = {
        K.RESULT: DATA_PARENT_DPATH,
        K.DATA: ['all'],
        'max_workers': 8,
    }

    def test(self):
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
