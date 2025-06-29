<!-- ============================================================
  Project Image
 ============================================================ -->
<!-- <div align=center>
  <img
    src='docs/image/demo.gif'
    alt='Project Image.'
    width=500
  />
</div> -->

<!-- ============================================================
  Overview
 ============================================================ -->

# :book:Overview

[![English](https://img.shields.io/badge/English-018EF5.svg?labelColor=d3d3d3&logo=readme)](./README.md)
[![Japanese](https://img.shields.io/badge/Japanese-018EF5.svg?labelColor=d3d3d3&logo=readme)](./README_JA.md)
[![license](https://img.shields.io/github/license/r-dev95/tensorflow-template)](./LICENSE)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[![Python](https://img.shields.io/badge/Python-3776AB.svg?labelColor=d3d3d3&logo=python)](https://github.com/python)
[![Sphinx](https://img.shields.io/badge/Sphinx-000000.svg?labelColor=d3d3d3&logo=sphinx&logoColor=000000)](https://github.com/sphinx-doc/sphinx)
[![Pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?labelColor=d3d3d3&logo=pytest)](https://github.com/pytest-dev/pytest)
[![Pydantic](https://img.shields.io/badge/Pydantic-ff0055.svg?labelColor=d3d3d3&logo=pydantic&logoColor=ff0055)](https://github.com/pydantic/pydantic)
[![Numpy](https://img.shields.io/badge/Numpy-blue.svg?labelColor=d3d3d3&logo=numpy&logoColor=blue)](https://github.com/numpy/numpy)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?labelColor=d3d3d3&logo=pandas&logoColor=150458)](https://github.com/pandas-dev/pandas)
[![Scipy](https://img.shields.io/badge/Scipy-0054a6.svg?labelColor=d3d3d3&logo=scipy&logoColor=0054a6)](https://github.com/scipy/scipy)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?labelColor=d3d3d3&logo=scikit-learn)](https://github.com/scikit-learn/scikit-learn)
[![Keras](https://img.shields.io/badge/Keras-D00000.svg?labelColor=d3d3d3&logo=keras&logoColor=D00000)](https://github.com/keras-team/keras)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?labelColor=d3d3d3&logo=tensorflow)](https://github.com/tensorflow/tensorflow)

This repository defines tensorflow templates.

By using a parameter file to set the following items, you can train and evaluate in various combinations.
You can also easily use other settings by implementing them in your own settings classes.

- data (and data preprocess)
- model (and model layer)
- optimizer method
- loss function
- metrics
- callbacks

We use Sphinx to create documentation for the implementation sources.

Please clone this repository and check it locally.

For more details about the build, please see [here](docs/sphinx_docs.md).

```bash
cd tensorflow-template/docs
make html -e SPHINXOPTS='-a -E -D language="en"' # "en" or "ja"
```

<!-- ============================================================
  Features
 ============================================================ -->
<!-- ## :desktop_computer:Features -->

<!-- ============================================================
  Usage
 ============================================================ -->

## :keyboard:Usage

### Install

```bash
git clone https://github.com/r-dev95/tensorflow-template.git
```

### Build virtual environment

You need to install `uv`.

If you don't have a python development environment yet, see [here](https://github.com/r-dev95/env-python).

```bash
cd tensorflow-template/
uv sync
```

### Download data and make tfrecord-form data

```bash
cd src
uv run python dataset.py --result dataset --data mnist
```

### Training the model

```bash
uv run python train.py --param param/tutorial/param_train.yaml
```

### Evaluate the model

```bash
uv run python eval.py --param param/tutorial/param_eval.yaml
```

## How to use parameter file

This section describes how to use parameter files (`.yaml`).

The parameter file is used in the following source code.
The following source code can use some command line arguments, but they can be overwritten in the parameter file, so it is assumed that all parameters are set in the parameter file.

- train.py
- eval.py

Some of the settings can not be set using parameter file. In particular, detailed settings for tensorflow (keras) must be implemented by referring to the official tensorflow (keras) website.

### Set the main parameters

Main parameters that are also implemented as command line arguments are set with zero indentation.

- The main parameters include `param`, but this is not set as it only works as a command line argument.

`train.py` and `eval.py` common settings example:

```yaml
# log handler (idx=0: stream handler, idx=1: file handler)
# (True: set handler, False: not set handler)
# type: list[bool, bool]
handler: [True, True]
# log level (idx=0: stream handler, idx=1: file handler)
# (DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50)
# type: list[int, int]
level: [10, 10]
# flag (eager mode: true, graph mode: false)
# type: boolean
eager: false
# random seed
# type: int
seed: 0
# directory path (data save)
# type: str
result: result
```

only `train.py` settings example:

```yaml
# directory path (training data)
# type: str
train: data/mnist/train
# directory path (validation data)
# type: str | None
valid: data/mnist/test
# batch size (training data)
# type: int
train_batch: 32
# batch size (validation data)
# type: int | None
valid_batch: 1000
# shuffle size
# type: int | None
shuffle: null
# Number of epochs
# type: int
epochs: 2
```

only `eval.py` settings example:

```yaml
# directory path (evaluation data)
# type: str
eval: data/mnist/test
# batch size (training data)
# type: int
batch: 1000
```

### Set the `data` and `data preprocess` parameters

For currently available `data`, see the variable `func`'s key of the `SetupData` class [here](tensorflow_template/lib/data/setup.py).

`data` settings example:

```yaml
data:
  kind: mnist
```

For currently available `data preprocess`, see the variable `func`'s key of the `Processor` class [here](tensorflow_template/lib/data/processor.py).

- The `kind` of `data preprocess` is set as a list.

- If you set `catencode` to `kind`, set `catencode` setting as shown in the following example.
  The same applies to the subsequent parameters.

`data preprocess` settings example:

```yaml
process:
  kind: [catencode, rescale]

  catencode:
    num_tokens: &num_classes 10
    output_mode: one_hot
    sparse: false

  rescale:
    scale: 0.00392156862745098
    offset: 0
```

### Set the `model` and `model layer` parameters

For currently available `model`, see the variable `func`'s key of the `SetupModel` class [here](tensorflow_template/lib/model/setup.py).

`model` settings example:

```yaml
model:
  kind: simple
```

For currently available `model layer`, see the variable `func`'s key of the `SetupLayer` class [here](tensorflow_template/lib/model/layer.py).

- The `kind` of `model layer` is set as a list.

- The value of `kind` can have "\_" + alphanumeric characters at the end.

`model layer` settings example:

```yaml
layer:
  kind: [flatten, dense_1, relu, dense_2]

  flatten:
    data_format: channels_last

  DENSE: &DENSE
    units: null
    activation: null
    use_bias: true
    kernel_initializer: glorot_uniform
    bias_initializer: zeros
    kernel_regularizer: null
    bias_regularizer: null
    activity_regularizer: null
    kernel_constraint: null
    bias_constraint: null
    lora_rank: null
  dense_1:
    <<: *DENSE
    units: 100
  dense_2:
    <<: *DENSE
    units: *num_classes

  relu:
    max_value: null
    negative_slope: 0
    threshold: 0
```

### Set the `optimizer method` parameters

For currently available `optimizer method`, see the variable `func`'s key of the `SetupOpt` class [here](tensorflow_template/lib/optimizer/setup.py).

- The `optimizer method` parameter is only used in `train.py`.

`optimizer method` settings example:

```yaml
opt:
  kind: adam

  adam:
    learning_rate: 0.001
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 0.0000001
    amsgrad: false
    weight_decay: null
    clipnorm: null
    clipvalue: null
    global_clipnorm: null
    use_ema: false
    ema_momentum: 0.99
    ema_overwrite_frequency: null
    loss_scale_factor: null
    gradient_accumulation_steps: null
    name: adam
```

### Set the `loss function` parameters

For currently available `loss function`, see the variable `func`'s key of the `SetupLoss` class [here](tensorflow_template/lib/loss/setup.py).

`loss function` settings example:

```yaml
loss:
  kind: cce

  cce:
    from_logits: true
    label_smoothing: 0
    axis: -1
    reduction: sum_over_batch_size
    name: categorical_crossentropy
    # dtype: null
```

### Set the `metrics` parameters

For currently available `metrics`, see the variable `func`'s key of the `SetupMetrics` class [here](tensorflow_template/lib/metrics/setup.py).

- The `kind` of `metrics` is set as a list.

`metrics` settings example:

```yaml
metrics:
  kind: [mse, cacc]

  mse:
    name: mean_squared_error
    # dtype: null

  cacc:
    name: categorical_accuracy
    # dtype: null
```

### Set the `callbacks` parameters

For currently available `callbacks`, see the variable `func`'s key of the `SetupCallbacks` class [here](tensorflow_template/lib/callbacks/setup.py).

- The `callbacks` parameter is only used in `train.py`.

- The `kind` of `callbacks` is set as a list.

`callbacks` settings example:

```yaml
cb:
  kind: [mcp, csv]

  mcp:
    # filepath: null            # The "filepath" is fixed in the code.
    monitor: val_loss
    verbose: 0
    save_best_only: false
    save_weights_only: true
    mode: auto
    save_freq: epoch
    initial_value_threshold: null

  csv:
    # filename: null            # The "filename" is fixed in the code.
    separator: ","
    append: false
```

<!-- ============================================================
  Structure
 ============================================================ -->

## :bookmark_tabs:Structure

<div align=center>
  <img
    src='docs/image/classes.svg'
    alt='classes.'
  />
</div>

<!-- ============================================================
  License
 ============================================================ -->

## :key:License

This repository is licensed under the [Apache License 2.0](LICENSE).
