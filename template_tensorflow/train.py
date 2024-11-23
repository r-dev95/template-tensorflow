"""This is the module that trains the model.
"""  # noqa: INP001

import argparse
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

from lib.callbacks.setup import SetupCallbacks
from lib.common.decorator import process_time, save_params_log
from lib.common.define import ParamKey, ParamLog
from lib.common.file import load_yaml
from lib.common.log import SetLogging
from lib.common.process import fix_random_seed
from lib.data.setup import SetupData
from lib.loss.setup import SetupLoss
from lib.metrics.setup import SetupMetrics
from lib.model.setup import SetupModel
from lib.optimizer.setup import SetupOpt

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:  # noqa: C901, PLR0912
    """Checks the :class:`Trainer` parameters.

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
    if (params[K.PARAM] is None) or (not Path(params[K.PARAM]).exists()):
        error = True
        LOGGER.error(f'params["{K.PARAM}"] is None or the file does not exists.')
    if (params[K.TRAIN] is None) or (not Path(params[K.TRAIN]).exists()):
        error = True
        LOGGER.error(f'params["{K.TRAIN}"] is None or the directory does not exists.')
    if (params[K.VALID] is None) or (not Path(params[K.VALID]).exists()):
        LOGGER.warning(f'params["{K.VALID}"] is None or the directory does not exists.')
        LOGGER.warning(f'Run without validation data.')
    if (params[K.RESULT] is None) or (not Path(params[K.RESULT]).exists()):
        error = True
        LOGGER.error(f'params["{K.RESULT}"] is None or the directory does not exists.')
    if (params[K.EPOCHS] is None) or (params[K.EPOCHS] <= 0):
        error = True
        LOGGER.error(f'params["{K.EPOCHS}"] must be greater than zero.')
    if (params[K.BATCH_TRAIN] is None) or (params[K.BATCH_TRAIN] <= 0):
        error = True
        LOGGER.error(f'params["{K.BATCH_TRAIN}"] must be greater than zero.')
    if (params[K.BATCH_VALID] is None) or (params[K.BATCH_VALID] <= 0):
        LOGGER.warning(f'params["{K.BATCH_VALID}"] must be greater than zero or None.')
    if params[K.SHUFFLE] is None:
        LOGGER.warning(f'params["{K.SHUFFLE}"] is None.')
        LOGGER.warning(f'The data is not shuffled.')

    keys = [K.DATA, K.PROCESS, K.MODEL, K.LAYER, K.OPT, K.LOSS, K.METRICS, K.CB]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class Trainer:
    """Trains the model.

    *   You can train in various combinations depending on the configuration of each
        class in the table below.
        If you want to use other configuration, implement them in the each functions.

        +------------------------------+---------------------------------------------+
        |class                         |function                                     |
        +==============================+=============================================+
        |data                          |:class:`lib.data.setup.SetupData`            |
        +------------------------------+---------------------------------------------+
        |optimizer method              |:class:`lib.optimizer.setup.SetupOpt`        |
        +------------------------------+---------------------------------------------+
        |loss function                 |:class:`lib.loss.setup.SetupLoss`            |
        +------------------------------+---------------------------------------------+
        |metrics                       |:class:`lib.metrics.setup.SetupMetrics`      |
        +------------------------------+---------------------------------------------+
        |callback                      |:class:`lib.callbacks.setup.SetupCallbacks`  |
        +------------------------------+---------------------------------------------+
        |model                         |:class:`lib.model.setup.SetupModel`          |
        +------------------------------+---------------------------------------------+

    Args:
        params (dict[str, Any]): parameters.
    """
    #: ClassVar[dict[str, Any]]: class list
    #:
    #: *    key=opt: optimizer method class
    #: *    key=loss: loss function class
    #: *    key=metrics: list of metrics classes
    classes: ClassVar[dict[str, Any]] = {}
    #: Callable: model class
    model: Callable
    #: list[Callable]: list of callback classes
    callbacks: list[Callable]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        check_params(params=params)
        fix_random_seed(seed=params[K.SEED])

        self.load_dataset()
        self.setup()

    def load_dataset(self) -> None:
        """Loads the training and validation data.

        *   The training data must be loaded, but the validation data does not
            necessarily have to be loaded.
        """
        # training data
        self.params[K.FPATH] = self.params[K.TRAIN]
        self.params[K.BATCH] = self.params[K.BATCH_TRAIN]
        self.train_data = SetupData(params=self.params).setup()
        self.train_loader = self.train_data.make_loader_example()
        self.train_steps_per_epoch = self.train_data.steps_per_epoch
        # validation data
        self.params[K.FPATH] = self.params[K.VALID]
        self.params[K.BATCH] = self.params[K.BATCH_VALID]
        self.params[K.SHUFFLE] = None
        self.valid_data = SetupData(params=self.params).setup()
        self.valid_loader = self.valid_data.make_loader_example()
        self.valid_steps_per_epoch = self.valid_data.steps_per_epoch

    def setup(self) -> None:
        """Sets up the training.

        *   Sets the optimizer method, loss function, model, metrics, and callbacks.
        *   Run ``.summary()``.
        """
        self.classes[K.OPT] = SetupOpt(params=self.params).setup()
        self.classes[K.LOSS] = SetupLoss(params=self.params).setup()
        self.classes[K.METRICS] = SetupMetrics(params=self.params).setup()
        self.callbacks = SetupCallbacks(params=self.params).setup()
        self.params[K.CLASSES] = self.classes
        self.params[K.INPUT_SHAPE] = self.train_data.input_shape_model
        self.model = SetupModel(params=self.params).setup()
        self.model.summary()

    def run(self) -> None:
        """Runs training and validation.

        *   Run ``.compile()`` and ``.fit()``.
        """
        self.model.compile(
            run_eagerly=self.params[K.EAGER],
        )
        self.model.fit(
            x=self.train_loader,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_data=self.valid_loader,
            validation_steps=self.valid_steps_per_epoch,
            epochs=self.params[K.EPOCHS],
            callbacks=self.callbacks,
            verbose=1,
        )


@save_params_log(fname=f'log_params_{Path(__file__).stem}.yaml')
@process_time(print_func=LOGGER.info)
def main(params: dict[str, Any]) -> dict[str, Any]:
    """main.

    This function is decorated by ``@save_params_log`` and ``@process_time``.

    Args:
        params (dict[str, Any]): parameters.

    Returns:
        dict[str, Any]: parameters.
    """
    train = Trainer(params=params)
    train.run()
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
    parser = argparse.ArgumentParser()
    # log level (idx=0: stream handler, idx=1: file handler)
    # (DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50)
    choices = [10, 20, 30, 40, 50]
    parser.add_argument('--level', default=[20, 20], type=int, nargs=2, choices=choices)
    # flag (eager mode: true, graph mode: false)
    parser.add_argument('--eager', default=False, action='store_true')
    # random seed
    parser.add_argument('--seed', default=0, type=int)
    # file path (parameters)
    parser.add_argument('--param', default='param/param.yaml', type=str)
    # directory path (data save)
    parser.add_argument('--result', default='result', type=str)
    # directory path (training data)
    parser.add_argument('--train', default='', type=str)
    # directory path (validation data)
    parser.add_argument('--valid', default='', type=str)
    # Number of epochs
    parser.add_argument('--epochs', default=10, type=int)
    # batch size (training data)
    parser.add_argument('--batch_train', default=32, type=int)
    # batch size (validation data)
    parser.add_argument('--batch_valid', default=1000, type=int)
    # shuffle size
    parser.add_argument('--shuffle', default=None, type=int)

    params = vars(parser.parse_args())

    # set the file parameters.
    fpath = Path(params[K.PARAM])
    if K.PARAM in params and fpath.is_file():
        params.update(load_yaml(fpath=fpath))

    return params


if __name__ == '__main__':
    # set the parameters.
    params = set_params()
    # set the logging configuration.
    PARAM_LOG.LEVEL[PARAM_LOG.SH] = params[K.LEVEL][0]
    PARAM_LOG.LEVEL[PARAM_LOG.FH] = params[K.LEVEL][1]
    SetLogging(logger=LOGGER, param=PARAM_LOG)

    if K.RESULT in params:
        Path(params[K.RESULT]).mkdir(parents=True, exist_ok=True)

    main(params=params)
