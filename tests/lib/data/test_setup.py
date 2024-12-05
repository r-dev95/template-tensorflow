"""This is the module that tests setup.py.
"""

import sys
from logging import ERROR, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import setup

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


MNIST         = {K.DATA: {K.KIND: 'mnist'}}
FASHION_MNIST = {K.DATA: {K.KIND: 'fashion_mnist'}}
CIFAR10       = {K.DATA: {K.KIND: 'cifar10'}}
CIFAR100      = {K.DATA: {K.KIND: 'cifar100'}}
_params = [MNIST, FASHION_MNIST, CIFAR10, CIFAR100]
for _param in _params:
    _param[K.DPATH] = 'train'
    _param[K.PROCESS] = {K.KIND: []}
    _param[K.BATCH] = 10
    _param[K.SHUFFLE] = None
    _param[K.REPEAT] = 1


class TestSetupData:
    """Tests :class:`setup.SetupData`.
    """

    @pytest.mark.parametrize(
            ('params', 'label'), [
                (MNIST        , "<class 'lib.data.mnist.Mnist'>"),
                (FASHION_MNIST, "<class 'lib.data.mnist.Mnist'>"),
                (CIFAR10      , "<class 'lib.data.cifar.Cifar'>"),
                (CIFAR100     , "<class 'lib.data.cifar.Cifar'>"),
            ],
    )
    def test(self, params, label):
        """Tests that no errors are raised.

        *   The class type is correct.
        """
        _class = setup.SetupData(params=params).setup()
        print(f'{type(_class)=}')
        assert str(type(_class)) == label

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.DATA: {
                K.KIND: '',
            },
        }
        with pytest.raises(ValueError):
            setup.SetupData(params=params).setup()

        all_log = [
            ('main', ERROR, f'SetupData class does not have a method "{params[K.DATA][K.KIND]}" that sets the data.'),
            ('main', ERROR, f'The available data are:'),
        ]
        func = setup.SetupData(params=MNIST).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
