"""This is the module that tests base.py.
"""

import sys
from logging import ERROR, INFO, getLogger

import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.model import base

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """
    classes = {
        K.OPT: '',
        K.LOSS: '',
        K.METRICS: '',
    }
    classes_raise = {}

    all_log = []
    for key in [K.OPT, K.LOSS, K.METRICS]:
        all_log.append(('main', ERROR  , f'The key "{key}" for variable "classes" is missing.'))

    def test(self):
        """Tests that no errors are raised.
        """
        base.check_params(classes=self.classes)

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        caplog.set_level(INFO)
        with pytest.raises(ValueError):
            base.check_params(classes=self.classes_raise)

        assert caplog.record_tuples == self.all_log


class TestBaseModel:
    """Tests :class:`base.BaseModel`.

    *   This class is a base class and is not called directly, so it will not be tested.
    *   It is tested by :class:`test_train.TestTrain`
        and :class:`test_eval.TestEval` class.
    """
