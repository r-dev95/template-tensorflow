"""This is the module that tests processor.py.
"""

import sys
from logging import ERROR, getLogger

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture

sys.path.append('../template_tensorflow/')
from template_tensorflow.lib.common.define import ParamKey, ParamLog
from template_tensorflow.lib.data import processor

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(name=PARAM_LOG.NAME)


CATENCODE = {
    'num_tokens': 10,
    'output_mode': 'one_hot',
    'sparse': False,
}
RESCALE = {
    'scale': 10,
    'offset': 0,
}


class TestProcessor:
    """Tests :class:`processor.Processor`.
    """

    def test_catencode(self):
        """Tests that no errors are raised.

        *   This class tests :class:`processor.Processor.catencode`.
        *   The process result is correct.
        """
        kind = 'catencode'
        params = {}
        params[K.PROCESS] = {K.KIND: [kind], kind: CATENCODE}
        _class = processor.Processor(params=params)

        x_ini = np.array([[1, 2], [3, 4]])
        y_ini = np.array([4, 9])
        x, y = _class.run(x=x_ini, y=y_ini)

        assert (x_ini == x).all()
        assert (np.identity(CATENCODE['num_tokens'])[y_ini] == y.numpy()).all()

    def test_rescale(self):
        """Tests that no errors are raised.

        *   This class tests :class:`processor.Processor.rescale`.
        *   The process result is correct.
        """
        kind = 'rescale'
        params = {}
        params[K.PROCESS] = {K.KIND: [kind], kind: RESCALE}
        _class = processor.Processor(params=params)

        x_ini = np.array([[1, 2], [3, 4]])
        y_ini = np.array([4, 9])
        x, y = _class.run(x=x_ini, y=y_ini)

        assert (x_ini * RESCALE['scale'] == x.numpy()).all()
        assert (y_ini == y).all()

    def test_raise(self, caplog: LogCaptureFixture):
        """Tests that an error is raised.

        *   The log output is correct.
        """
        params = {
            K.PROCESS: {
                K.KIND: [''],
                '': {},
            },
        }
        with pytest.raises(ValueError):
            processor.Processor(params=params)

        all_log = [
            ('main', ERROR, f'Processor class does not have a method "{params[K.PROCESS][K.KIND][0]}" that sets the processing method.'),
            ('main', ERROR, f'The available processing method are:'),
        ]
        kind = 'catencode'
        params[K.PROCESS] = {K.KIND: [kind], kind: CATENCODE}
        func = processor.Processor(params=params).func
        print(f'{func=}')
        for key in func:
            all_log.append(('main', ERROR, f'{key=}'))

        assert caplog.record_tuples == all_log
