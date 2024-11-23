"""This is the module that tests cifar.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import cifar

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCifar:
    """Tests :class:`cifar.Cifar`.
    """
    params = {
        K.RESULT: 'result',
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1000,
    }

    @pytest.fixture(scope='class')
    def proc(self):
        fpath = Path(self.params[K.RESULT])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        params = {
            K.RESULT: self.params[K.RESULT],
            'max_workers': 8,
        }
        params[K.DATA] = ['cifar10', 'cifar100']
        dataset.main(params=params)
        yield
        shutil.rmtree(fpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   Data can load correctly. (cifar10, cifar100)
        """
        # cifar10
        self.params[K.DATA] = {K.KIND: 'cifar10'}
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'cifar10', 'train')
        data = cifar.Cifar(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rcifar10-train: {inputs.numpy().shape=}, {labels.numpy().shape=}')
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'cifar10', 'test')
        data = cifar.Cifar(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rcifar10-test: {inputs.numpy().shape=}, {labels.numpy().shape=}')

        # cifar100
        self.params[K.DATA] = {K.KIND: 'cifar100'}
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'cifar100', 'train')
        data = cifar.Cifar(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rcifar100-train: {inputs.numpy().shape=}, {labels.numpy().shape=}')
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'cifar100', 'test')
        data = cifar.Cifar(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rcifar100-test: {inputs.numpy().shape=}, {labels.numpy().shape=}')
