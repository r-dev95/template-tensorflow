"""This is the module load and write data.
"""

from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
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
    params[K.REPEAT] = params.get(K.EPOCHS, 1)
    params[K.SHUFFLE] = params.get(K.SHUFFLE)


class BaseLoadData:
    """Loads data.

    *   Make a data pipeline to load a TFRecord data.

    Args:
        params (dict[str, Any]): parameters.

    .. attention::

        Child classes that inherit this class must set the list of file paths to
        ``params[K.FILES]`` before running ``super().__init__(params=params)``.
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

    def preprocess(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs preprocess.

        *   Run :meth:`lib.data.processor.Processor.run`.

        Args:
            x (tf.Tensor): input. (before preprocess)
            y (tf.Tensor): label. (before preprocess)

        Returns:
            tf.Tensor: input. (after preprocess)
            tf.Tensor: label. (after preprocess)
        """
        if self.params[K.PROCESS][K.KIND] is not None:
            x, y = self.Processor.run(x=x, y=y)
        # x = tf.reshape(x, self.input_shape_model)
        # y = tf.reshape(y, self.label_shape_model)
        return x, y

    def parse_example(self, example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Parses one example from a TFRecord data.

        #.  Set the parsing configuration according to the format in which the data was
            written. (``tf.io.parse_single_example()``)
        #.  When writing TFRecord data, we make the elements one-dimensional, so we
            restore the shape.
        #.  Run preprocess. (:meth:`preprocess`)

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
        x, y = self.preprocess(x=x, y=y)
        return x, y

    def make_loader_example(self, seed: int = 0) -> Callable:
        """Makes data loader.

        #.  Set the list of data file pathes.
            (``tf.data.Dataset.list_files()``)
        #.  Set the interleave configuration.
            (``tf.data.Dataset.interleave()``)
        #.  Set the function to parse one example from a TFRecord data.
            (``tf.data.Dataset.map()``)
        #.  Set the shuffle configuration.
            (``tf.data.Dataset.shuffle()``)
        #.  Set the batch configuration.
            (``tf.data.Dataset.batch()``)
        #.  Set the prefetch configuration.
            (``tf.data.Dataset.prefetch()``)
        #.  Set the repeat configuration.
            (``tf.data.Dataset.repeat()``)

        Args:
            seed (int): random seed.

        Returns:
            Callable: data pipeline. (``tf.data``)
        """
        dataset = tf.data.Dataset.list_files(file_pattern=self.params[K.FILES])
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


def feature_int64_list(value: list[int]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.Int64List``)

    Args:
        value (list[int]): one-dimensional list with elements of type ``int``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [1,2,3]
        print(feature_int64_list(value=value))

        # int64_list {
        # value: 1
        # value: 2
        # value: 3
        # }
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def feature_float_list(value: list[float]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.FloatList``)

    Args:
        value (list[float]): one-dimensional list with elements of type ``float``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [1.,2.,3.]
        print(feature_float_list(value=value))

        # float_list {
        # value: 1
        # value: 2
        # value: 3
        # }
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def feature_bytes_list(value: list[bytes]) -> tf.train.Feature:
    """Converts ``tf.train.Feature`` type. (``tf.train.BytesList``)

    Args:
        value (list[bytes]): one-dimensional list with elements of type ``bytes``.

    Returns:
        tf.train.Feature: value of ``tf.train.Feature`` type.

    .. code-block:: python

        value = [b"1", b"test"]
        print(feature_bytes_list(value=value))

        # bytes_list {
        # value: "1"
        # value: "test"
        # }
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def serialize_example(data: list[list[float]]) -> tf.train.Example:
    """Converts ``tf.train.Example`` type.

    Args:
        data (list[float]): a set of inputs and labels.

    Returns:
        tf.train.Example: value of ``tf.train.Example`` type.
    """
    feature = {
        'input': feature_float_list(data[0]),
        'label': feature_float_list(data[1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_exmaple(fpath: Path, inputs: np.ndarray, labels: np.ndarray) -> None:
    """Writes TFRecord data.

    Args:
        fpath (Path): file path.
        inputs (np.ndarray): input.
        labels (np.ndarray): label.
    """
    inputs = np.squeeze(inputs)
    labels = np.squeeze(labels)

    writer = tf.io.TFRecordWriter(fpath.as_posix())
    for i, (x, y) in enumerate(zip(inputs, labels)):
        example_proto = serialize_example([x.flatten(), [y]])
        writer.write(example_proto.SerializeToString())
        print(f'\rsave progress: {i + 1:>7} / {len(inputs)}', end='')
    print()
    writer.close()
