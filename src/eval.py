"""This is the module that evaluates the model.
"""  # noqa: INP001

import argparse
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import keras
import numpy as np

from lib.common.decorator import process_time, save_params_log
from lib.common.file import load_yaml
from lib.common.log import SetLogging
from lib.common.process import fix_random_seed, set_weight
from lib.common.types import ParamKey as K
from lib.common.types import ParamLog
from lib.data.base import BaseLoadData
from lib.data.setup import SetupData
from lib.loss.setup import SetupLoss
from lib.metrics.setup import SetupMetrics
from lib.model.base import BaseModel
from lib.model.setup import SetupModel

PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:  # noqa: C901
    """Checks the :class:`Evaluator` parameters.

    Args:
        params (dict[str, Any]): parameters.
    """
    error = False # error: True
    if not isinstance(params[K.EAGER], bool):
        error = True
        LOGGER.error(f'params["{K.EAGER}"] must be boolean.')
    if not isinstance(params[K.SEED], int):
        LOGGER.warning(f'params["{K.SEED}"] must be integer.')
        LOGGER.warning(f'The random number seed is not fixed.')
    if (params[K.RESULT] is None) or (not Path(params[K.RESULT]).exists()):
        error = True
        LOGGER.error(f'params["{K.RESULT}"] is None or the directory does not exists.')
    if (params[K.EVAL] is None) or (not Path(params[K.EVAL]).exists()):
        error = True
        LOGGER.error(f'params["{K.EVAL}"] is None or the directory does not exists.')
    if (params[K.BATCH] is None) or (params[K.BATCH] <= 0):
        error = True
        LOGGER.error(f'params["{K.BATCH}"] must be greater than zero.')

    keys = [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.LOSS, K.METRICS]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class Evaluator:
    """Evaluates the model.

    Args:
        params (dict[str, Any]): parameters.
    """
    #: BaseLoadData: data class (evaluate)
    eval_data: BaseLoadData
    #: ClassVar[dict[str, Any]]: class list
    #:
    #: *    key=opt: optimizer method class
    #: *    key=loss: loss function class
    #: *    key=metrics: list of metrics classes
    classes: ClassVar[dict[str, Any]] = {}
    #: BaseModel: model class
    model: BaseModel
    #: list[Callable]: list of callback classes
    callbacks: list[Callable]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        check_params(params=params)
        fix_random_seed(seed=params[K.SEED])

        self.load_dataset()
        self.setup()

    def load_dataset(self) -> None:
        """Loads the evaluation data.
        """
        # evaluation data
        self.params[K.DPATH] = self.params[K.EVAL]
        self.params[K.SHUFFLE] = None
        self.params[K.REPEAT] = 1
        self.eval_data = SetupData(params=self.params).setup()

    def setup(self) -> None:
        """Sets up the evaluation.

        *   Sets the loss function, model, metrics.
        *   Set the model weights.
        *   Run ``.summary``.
        """
        self.classes[K.OPT] = None
        self.classes[K.LOSS] = SetupLoss(params=self.params).setup()
        self.classes[K.METRICS] = SetupMetrics(params=self.params).setup()
        self.params[K.CLASSES] = self.classes
        self.params[K.INPUT_SHAPE] = self.eval_data.input_shape_model
        self.model = SetupModel(params=self.params).setup()
        self.model = set_weight(params=self.params, model=self.model)
        self.model.summary()

    def eval_step(self) -> dict[str, Any]:
        """Evaluations the model.

        *   Customize the evaluation of your trained models.

        Returns:
            dict[str, Any]: evaluate results.
        """
        i_data = 0
        n_data = self.eval_data.n_data
        for inputs, labels in self.eval_data.make_loader_example():
            i_data += len(inputs)
            preds = self.model(inputs)
            losses = self.classes[K.LOSS](labels, preds)

            res = self.model.update_metrics(data=(labels, preds, losses, None))

            msg = f'\r[{K.EVAL}][{i_data:>8} / {n_data:>8}] - '
            for key, val in res.items():
                msg += f'{key}={val:>.5}, '
            print(msg, end='')
        print()

        res = {}
        for m in self.classes[K.METRICS]:
            res[m.name] = m.result()
        return res

    def run(self) -> None:
        """Runs evaluation.

        *   Run ``.compile``.
        *   Customize the evaluation of your trained models.
        """
        self.model.compile(
            run_eagerly=self.params[K.EAGER],
        )

        res = self.eval_step()

        msg = ''
        for key, val in res.items():
            msg += f'{key}={val:>.5}, '
        LOGGER.info(msg)


@save_params_log(fname=f'log_params_{Path(__file__).stem}.yaml')
@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> dict[str, Any]:
    """Main.

    This function is decorated by ``@save_params_log`` and ``@process_time``.

    Args:
        params (dict[str, Any]): parameters.

    Returns:
        dict[str, Any]: parameters.
    """
    evaluate = Evaluator(params=params)
    evaluate.run()
    return params


def set_params() -> dict[str, Any]:
    """Sets the command line arguments and file parameters.

    *   Set only common parameters as command line arguments.
    *   Other necessary parameters are set in the file parameters.
    *   Use a yaml file. (:func:`lib.common.file.load_yaml`)

    Returns:
        dict[str, Any]: parameters.

    .. attention::

        Command line arguments are overridden by file parameters.
        This means that if you want to set everything using file parameters,
        you don't necessarily need to use command line arguments.
    """
    # set the command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        f'--{K.HANDLER}',
        default=[True, True], type=bool, nargs=2,
        help=(
            f'The log handler flag to use.\n'
            f'True: set handler, False: not set handler\n'
            f'ex) --{K.HANDLER} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.LEVEL}',
        default=[20, 20], type=int, nargs=2, choices=[10, 20, 30, 40, 50],
        help=(
            f'The log level.\n'
            f'DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50\n'
            f'ex) --{K.LEVEL} arg1 arg2 (arg1: stream handler, arg2: file handler)'
        ),
    )
    parser.add_argument(
        f'--{K.PARAM}',
        default='param/param.yaml', type=str,
        help=('The parameter file path.'),
    )
    parser.add_argument(
        f'--{K.RESULT}',
        default='result', type=str,
        help=('The directory path to save the results.'),
    )
    parser.add_argument(
        f'--{K.EAGER}',
        default=False, action='store_true',
        help=(
            'The running mode flag.\n'
            'eager mode: true, graph mode: false'
        ),
    )
    parser.add_argument(
        f'--{K.SEED}',
        default=0, type=int,
        help=('The random seed.'),
    )
    parser.add_argument(
        f'--{K.EVAL}',
        default='', type=str,
        help=('The evaluation data directory path.'),
    )
    parser.add_argument(
        f'--{K.BATCH}',
        default=1000, type=int,
        help=('The evaluation batch size.'),
    )

    params = vars(parser.parse_args())

    # set the file parameters.
    if params.get(K.PARAM):
        fpath = Path(params[K.PARAM])
        if fpath.is_file():
            params.update(load_yaml(fpath=fpath))

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.HANDLER[PARAM_LOG.SH] = params[K.HANDLER][0]
    PARAM_LOG.HANDLER[PARAM_LOG.FH] = params[K.HANDLER][1]
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if params.get(K.RESULT):
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)
