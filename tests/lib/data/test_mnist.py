"""This is the module that tests mnist.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import mnist

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestMnist:
    """Tests :class:`mnist.Mnist`.
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
        params[K.DATA] = ['mnist', 'fashion_mnist']
        dataset.main(params=params)
        yield
        shutil.rmtree(fpath)

    def test(self, proc):
        """Tests that no errors are raised.

        *   Data can load correctly. (mnist, fashion_mnist)
        """
        # mnist
        self.params[K.DATA] = {K.KIND: 'mnist'}
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'mnist', 'train')
        data = mnist.Mnist(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rmnist-train: {inputs.numpy().shape=}, {labels.numpy().shape=}', end='')
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'mnist', 'test')
        data = mnist.Mnist(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rmnist-test: {inputs.numpy().shape=}, {labels.numpy().shape=}', end='')

        # fashion_mnist
        self.params[K.DATA] = {K.KIND: 'fashion_mnist'}
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'fashion_mnist', 'train')
        data = mnist.Mnist(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rfashion_mnist-train: {inputs.numpy().shape=}, {labels.numpy().shape=}', end='')
        self.params[K.FPATH] = Path(self.params[K.RESULT], 'fashion_mnist', 'test')
        data = mnist.Mnist(params=self.params)
        loader = data.make_loader_example()
        for inputs, labels in loader:
            print(f'\rfashion_mnist-test: {inputs.numpy().shape=}, {labels.numpy().shape=}', end='')
