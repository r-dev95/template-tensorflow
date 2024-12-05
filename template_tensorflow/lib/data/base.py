"""This is the module load data.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Any, ClassVar

import tensorflow as tf

from lib.common.define import ParamKey, ParamLog
from lib.data.processor import Processor

K = ParamKey()
PARAM_LOG = ParamLog()
LOGGER = getLogger(PARAM_LOG.NAME)


def check_params(params: dict[str, Any]) -> None:
    """Checks the :class:`BaseLoadData` parameters.

    Args:
        params (dict[str, Any]): parameters.
    """
    error = False # error: True
    keys = [K.FILE_PATTERN, K.BATCH, K.SHUFFLE, K.REPEAT]
    for key in keys:
        if key not in params:
            error = True
            LOGGER.error(f'The key "{key}" for variable "params" is missing.')
    if error:
        raise ValueError


class BaseLoadData:
    """Loads data.

    *   Make a data pipeline to load a TFRecord data.

    Args:
        params (dict[str, Any]): parameters.

    .. attention::

        Child classes that inherit this class must set the pattern of file paths to
        ``params[K.FILE_PATTERN]`` before running ``super().__init__(params=params)``.
    """
    #: int: all number of data.
    n_data: int
    #: int: steps per epoch.
    steps_per_epoch: int
    #: int: input size. (elements per input)
    input_size: int
    #: int: label size. (elements per label)
    label_size: int
    #: ClassVar[list[int]]: input shape. (before preprocess)
    input_shape: ClassVar[list[int]]
    #: ClassVar[list[int]]: label shape. (before preprocess)
    label_shape: ClassVar[list[int]]
    #: ClassVar[list[int]]: input shape. (after preprocess)
    input_shape_model: ClassVar[list[int]]
    #: ClassVar[list[int]]: label shape. (after preprocess)
    label_shape_model: ClassVar[list[int]]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        check_params(params=params)

        self.Processor = Processor(params=params)
        self.set_model_il_shape()

        self.steps_per_epoch = (self.n_data - 1) // params[K.BATCH] + 1

    def set_model_il_shape(self) -> None:
        """Sets the shape of the preprocessed inputs and labels.
        """
        raise NotImplementedError

    def process(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs process data.

        *   Run :meth:`lib.data.processor.Processor.run`.

        Args:
            x (tf.Tensor): input. (before process)
            y (tf.Tensor): label. (before process)

        Returns:
            tf.Tensor: input. (after process)
            tf.Tensor: label. (after process)
        """
        if self.params[K.PROCESS][K.KIND] is not None:
            x, y = self.Processor.run(x=x, y=y)
        # x = tf.reshape(x, self.input_shape_model)
        # y = tf.reshape(y, self.label_shape_model)
        return x, y

    def parse_example(self, example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Parses one example from a TFRecord data.

        #.  Set the parsing configuration according to the format in which the data was
            written. (``tf.io.parse_single_example``)
        #.  When writing TFRecord data, we make the elements one-dimensional, so we
            restore the shape.
        #.  Run :meth:`process`.

        Args:
            example_proto (tf.Tensor): protocol massage.

        Returns:
            tf.Tensor: input.
            tf.Tensor: label.
        """
        features = {
            'input': tf.io.FixedLenFeature(self.input_size, dtype=tf.float32),
            'label': tf.io.FixedLenFeature(self.label_size, dtype=tf.float32),
        }
        example = tf.io.parse_single_example(
            serialized=example_proto,
            features=features,
        )
        x = tf.reshape(example['input'], self.input_shape)
        y = tf.reshape(example['label'], self.label_shape)
        x, y = self.process(x=x, y=y)
        return x, y

    def make_loader_example(self, seed: int = 0) -> Callable:
        """Makes data loader.

        #.  Set the file path pattern.
            (``tf.data.Dataset.list_files``)
        #.  Set the interleave configuration.
            (``tf.data.Dataset.interleave``)
        #.  Set the function to parse one example from a TFRecord data.
            (``tf.data.Dataset.map``)
        #.  Set the shuffle configuration.
            (``tf.data.Dataset.shuffle``)
        #.  Set the batch configuration.
            (``tf.data.Dataset.batch``)
        #.  Set the prefetch configuration.
            (``tf.data.Dataset.prefetch``)
        #.  Set the repeat configuration.
            (``tf.data.Dataset.repeat``)

        Args:
            seed (int): random seed.

        Returns:
            Callable: data pipeline. (``tf.data``)
        """
        dataset = tf.data.Dataset.list_files(file_pattern=self.params[K.FILE_PATTERN])
        dataset = dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(self.parse_example)
        if self.params[K.SHUFFLE] is not None:
            dataset = dataset.shuffle(
                buffer_size=self.params[K.SHUFFLE],
                seed=seed,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.batch(batch_size=self.params[K.BATCH])
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(count=self.params[K.REPEAT])
        return dataset
