"""This is the module that tests eval.py.
"""

import shutil
import sys
from logging import ERROR, INFO, WARNING
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow import dataset
from template_tensorflow.lib.common.define import ParamKey
from template_tensorflow.lib.data import base

K = ParamKey()


class TestFeature:
    """Tests the following functions.

    *   :func:`base.feature_bytes_list`
    *   :func:`base.feature_int64_list`
    *   :func:`base.feature_float_list`
    *   :func:`base.serialize_example`
    """

    def test(self):
        """Tests that no errors are raised.
        """
        # feature_bytes_list
        value = [b'1', b'test']
        label = 'bytes_list {\n  value: "1"\n  value: "test"\n}\n'
        data = dataset.feature_bytes_list(value=value)
        assert label == str(data)

        # feature_int64_list
        value = [1, 2]
        label = 'int64_list {\n  value: 1\n  value: 2\n}\n'
        data = dataset.feature_int64_list(value=value)
        assert label == str(data)

        # feature_float_list
        value = [1.2, 3.4]
        label = 'float_list {\n  value: 1.2\n  value: 3.4\n}\n'
        data = dataset.feature_float_list(value=value)
        assert label == str(data)

        # serialize_example
        value = [[1, 2], [3, 4]]
        label = (
            'features {\n'
            '  feature {\n'
            '    key: "label"\n    value {\n      float_list {\n'
            '        value: 3\n        value: 4\n      }\n    }\n  }\n'
            '  feature {\n'
            '    key: "input"\n    value {\n      float_list {\n'
            '        value: 1\n        value: 2\n      }\n    }\n  }\n}\n'
        )
        data = dataset.serialize_example(data=value)
        assert label == str(data)


class TestWriteExmaple:
    """Tests :func:`base.write_exmaple`.
    """
    params = {
        K.PROCESS: {
            K.KIND: [],
        },
        K.BATCH: 1,
        K.FILES: ['result/tests.tfr'],
    }

    @pytest.fixture(scope='class')
    def proc(self):
        base.BaseLoadData.n_data = 2
        base.BaseLoadData.input_size = 2
        base.BaseLoadData.label_size = 1
        base.BaseLoadData.input_shape = [2]
        base.BaseLoadData.label_shape = [1]

        fpath = Path(self.params[K.FILES][0])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        inputs = [[1., 2.], [3., 4.]]
        labels = [0., 1.]
        yield fpath, inputs, labels
        shutil.rmtree(fpath.parent)

    def test(self, proc, mocker: MockerFixture):
        """Tests that no errors are raised.

        *   Data can be written to the file and load correctly.
        *   The :class:`base.BaseLoadData` is also being tested simultaneously in
            loading tests.
        """
        fpath, inputs, labels = proc
        dataset.write_exmaple(fpath=fpath, inputs=inputs, labels=labels)

        mocker.patch.object(base.BaseLoadData, 'set_model_il_shape', return_value=None)
        loader = base.BaseLoadData(params=self.params).make_loader_example()
        for i, (x, y) in enumerate(loader):
            assert (inputs[i] == x.numpy()).all()
            assert (labels[i] == y.numpy()).all()


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
