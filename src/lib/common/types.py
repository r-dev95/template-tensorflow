"""This is the module that defines the types.
"""

import enum
import logging
import zoneinfo
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

#: ZoneInfo class.
ZoneInfo = zoneinfo.ZoneInfo(key='Asia/Tokyo')


class ParamKey(enum.StrEnum):
    """Defines the dictionary key for the main parameters.
    """
    HANDLER = enum.auto()
    LEVEL = enum.auto()
    PARAM = enum.auto()
    RESULT = enum.auto()

    EAGER = enum.auto()
    SEED = enum.auto()
    TRAIN = enum.auto()
    VALID = enum.auto()
    EVAL = enum.auto()
    BATCH = enum.auto()
    BATCH_TRAIN = enum.auto()
    BATCH_VALID = enum.auto()
    SHUFFLE = enum.auto()
    REPEAT = enum.auto()
    EPOCHS = enum.auto()

    DATA = enum.auto()
    PROCESS = enum.auto()
    MODEL = enum.auto()
    LAYER = enum.auto()
    OPT = enum.auto()
    LOSS = enum.auto()
    METRICS = enum.auto()
    CB = enum.auto()
    KIND = enum.auto()

    DPATH = enum.auto()
    FILE_PATTERN = enum.auto()
    CLASSES = enum.auto()
    INPUT_SHAPE = enum.auto()

    MAX_WORKERS = enum.auto()


class ParamLog(BaseModel):
    """Defines the parameters used in the logging configuration.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
    )
    #: str: The stream handler key.
    SH: str = Field(default='sh', frozen=True)
    #: str: The file handler key.
    FH: str = Field(default='fh', frozen=True)
    #: str: The name to pass to ``logging.getLogger``.
    NAME: str = Field(default='main')
    #: ClassVar[dict[str, bool]]: The handler flag to use.
    HANDLER: ClassVar[dict[str, bool]] = {
        'sh': True,
        'fh': True,
    }
    #: ClassVar[dict[str, int]]: The log level.
    LEVEL: ClassVar[dict[str, int]] = {
        'sh': logging.DEBUG,
        'fh': logging.DEBUG,
    }
    #: str: The file path.
    FPATH: str = Field(default='log/log.txt')
    #: int: The max file size.
    SIZE: int = Field(default=int(1e+6), gt=0)
    #: int: The number of files.
    NUM: int = Field(default=10, gt=0)


class ParamFileName(BaseModel):
    """Defines the result file name.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
    )
    LOSS: str = Field(default='log_loss.csv')
    WIGHT: str = Field(
        default=('epoch{epoch:03d}_loss{loss:.3f}_{val_loss:.3f}.weights.h5'),
    )
