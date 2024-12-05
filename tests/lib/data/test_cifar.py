"""This is the module that tests cifar.py.
"""

import sys
from logging import getLogger
from pathlib import Path

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import cifar

sys.path.append('../tests')
from define import DATA_PARENT_DPATH

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCifar:
    """Tests :class:`cifar.Cifar`.
    """
    params = {
        K.RESULT: DATA_PARENT_DPATH,
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1000,
        K.SHUFFLE: None,
        K.REPEAT: 1,
    }

    def test(self):
        """Tests that no errors are raised.

        *   Data can load correctly. (cifar10, cifar100)
        """

        def _func(kind: str):
            self.params[K.DATA] = {K.KIND: kind}
            self.params[K.DPATH] = Path(self.params[K.RESULT], kind, 'train')
            data = cifar.Cifar(params=self.params)
            loader = data.make_loader_example()
            for inputs, labels in loader:
                print(f'\r{kind}-train: {inputs.numpy().shape=}, {labels.numpy().shape=}')
            self.params[K.DPATH] = Path(self.params[K.RESULT], kind, 'test')
            data = cifar.Cifar(params=self.params)
            loader = data.make_loader_example()
            for inputs, labels in loader:
                print(f'\r{kind}-test: {inputs.numpy().shape=}, {labels.numpy().shape=}')

        # cifar10
        _func(kind='cifar10')
        # cifar100
        _func(kind='cifar100')
