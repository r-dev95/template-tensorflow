"""This is the module that tests decorator.py.
"""

import re
import shutil
import sys
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path

import pytest
from _pytest.capture import MultiCapture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common import decorator
from template_tensorflow.lib.common.define import ParamKey, ParamLog

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


class TestProcessTime:
    """Tests :func:`decorator.process_time`.
    """

    def test(self, capfd: MultiCapture):
        """Tests that no errors are raised.

        *   The print output is correct.
        """

        @decorator.process_time(print_func=print)
        def func():
            return

        func()

        ptns = [
            r'^# \[START\] ========================================$',
            r'^# ================================================$',
            r'^# \d*.\d*e-\d*sec|\d*.\d*sec$',
            r'^# \[END\] ==========================================$',
        ]

        outs, errs = capfd.readouterr()
        assert not errs

        for out, ptn in zip(outs.split('\n'), ptns):
            assert re.match(pattern=ptn, string=out)


class TestSaveParamsLog:
    """Tests :func:`decorator.save_params_log`.
    """
    params = {
        K.RESULT: 'result',
    }

    @pytest.mark.parametrize(
            'fname', [
                ('log_params.yml'),
                ('log_params.yaml'),
                ('log_params.json'),
                ('log_params.toml'),
            ],
    )
    def test(self, fname: Path):
        """Tests that no errors are raised.

        *   A file is output.
        """

        @decorator.save_params_log(fname=fname)
        def func():
            return self.params

        func()

        assert Path(self.params[K.RESULT], fname).is_file()
        shutil.rmtree(self.params[K.RESULT])
