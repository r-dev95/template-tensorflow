"""This is the module that tests base.py.
"""

import sys
from logging import ERROR, INFO, WARNING, getLogger

import pytest

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import base

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestCheckParams:
    """Tests :func:`base.check_params`.
    """

    def test(self):
        """Tests that no errors are raised.

        *   Certain parameters are set.
        """
        params = {}
        base.check_params(params=params)

        assert K.REPEAT in params
        assert K.SHUFFLE in params


class TestBaseLoadData:
    """Tests :class:`base.BaseLoadData`.

    *   This class is a base class and is not called directly, so it will not be tested.
    *   It is tested by :class:`TestWriteExmaple` class.
    """
