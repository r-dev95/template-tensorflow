"""This is the module that tests base.py.
"""

import sys
from logging import ERROR, INFO, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import base

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    params = {
        K.FILE_PATTERN: ['result/tests.tfr'],
        K.BATCH: 1000,
        K.SHUFFLE: None,
        K.REPEAT: 1,
    }
    params_raise = {}

    all_log = []
    for key in [K.FILE_PATTERN, K.BATCH, K.SHUFFLE, K.REPEAT]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "params" is missing.'))

    def test(self):
        """Tests that no errors are raised.

        *   Certain parameters are set.
        """
        base.check_params(params=self.params)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            base.check_params(params=self.params_raise)

        assert caplog.record_tuples == self.all_log


class TestBaseLoadData:
    """Tests :class:`base.BaseLoadData`.

    *   This class is a base class and is not called directly, so it will not be tested.
    *   It is tested by :class:`TestWriteExmaple` class.
    """
