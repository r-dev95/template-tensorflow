"""This is the module that tests mnist.py.
"""

import sys
from logging import getLogger
from pathlib import Path

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import mnist

sys.path.append('../tests')
from define import DATA_PARENT_DPATH

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestMnist:
    """Tests :class:`mnist.Mnist`.
    """
    params = {
        K.RESULT: DATA_PARENT_DPATH,
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1000,
    }

    def test(self):
        """Tests that no errors are raised.

        *   Data can load correctly. (mnist, fashion_mnist)
        """
        def _func(kind: str):
            self.params[K.DATA] = {K.KIND: kind}
            self.params[K.FPATH] = Path(self.params[K.RESULT], kind, 'train')
            data = mnist.Mnist(params=self.params)
            loader = data.make_loader_example()
            for inputs, labels in loader:
                print(f'\r{kind}-train: {inputs.numpy().shape=}, {labels.numpy().shape=}')
            self.params[K.FPATH] = Path(self.params[K.RESULT], kind, 'test')
            data = mnist.Mnist(params=self.params)
            loader = data.make_loader_example()
            for inputs, labels in loader:
                print(f'\r{kind}-test: {inputs.numpy().shape=}, {labels.numpy().shape=}')

        # mnist
        _func(kind='mnist')
        # fashion_mnist
        _func(kind='fashion_mnist')
