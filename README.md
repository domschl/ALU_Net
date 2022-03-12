# ALU_Net

<a href="https://colab.research.google.com/github/domschl/ALU_Net/blob/main/ALU_Net.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

An ALU (arithmetic logic unit) via neural nets using Colab TPUs, GPUs or local hardware and local Jupyter instances.
If used with Colab, user's Google Drive is used for persistant storage.

It can be used with Mac M1 GPU, if [Apple's tensorflow plugin](https://developer.apple.com/metal/tensorflow-plugin/) is installed.

## Experiment

<img align="right" width="300" src="https://github.com/domschl/ALU_Net/blob/main/resources/ALU.png">

The neural network is expected to learn arithmetic and logic operations between two unsigned `bit_count - 1` bit integers. The possible operations are: `+`, `-`, `/` (integer division), `*`, `%` (modulo), `AND` boolean bitwise AND, `OR` boolean bitwise OR, '`XOR` boolean bitwise XOR and the comparators `=`, `<`, `>`, `!=`.
Each integer is encoded as `bit_count` input neurons [only `bit_count - 1` bits are currently used, positive ints only -- this might change] (`0.0` level for bit zero, `1.0` level for bit one), the operation is encoded binary as `op_count` (number of different ops, e.g. 12) neurons (one-hot encoding for the `op_count` opcodes). For multiplication, the operands are restricted to `2**(bit_count//2) - 1`, so that the result always fits into `2**bit_count`.
The result of the network is a `bit_count` bit integer again binary encoded. The value `True` is encoded as `2**bit_count - 1`, `False` is `0x0`, the output has dimension `bit_count`.

### Input modes

If `vector=False` in notebook, a linear input is expected: in this case, input vector has dimension (`2*bit_count + op_count`).
If `vector=True`, the input is encoded as three 'words': the first operand, the operation, and the second operand. All three are padded to size `ALU_Dataset.embedding_size`, and can be treated with NLP methods like [`SelfAttention`](https://github.com/domschl/ALU_Net/blob/4a0217353dfe40501e821896737fef3a3e3b1a96/keras_custom_layers.py#L143) (see [`keras_custom_layers.py`](https://github.com/domschl/ALU_Net/blob/main/keras_custom_layers.py)). Since Attention doesn't know about the order of tokens, three bits are appended for each of the three input vectors, the index of the vector is one-hot encoded in those three bits. That way attention can discriminate between op1, operation and op2.

### Example results after a short period of training

This shows lines in format:

```
<int> <operation> <int> = <net-eval> ==/!= <actual result> OK/Error
```

In case of an error, a binary representation of expected and actual result are shown.
The most difficult operation for the net to learn seems to be multiplication.
```
# bit_count = 15
16955 OR 15381 = 32319 == 32319: OK
17973 < 15428 = False == False: OK
2742 + 2691 = 5433 == 5433: OK
43 * 107 = 4601 != 4537: Error
0b1000110111001
0b1000111111001
21271 / 9519 = 2 == 2: OK
23480 / 27797 = 0 == 0: OK
10283 XOR 30755 = 20488 == 20488: OK
25079 - 6199 = 18880 == 18880: OK
18646 XOR 24019 = 5381 == 5381: OK
28916 + 24293 = 53209 == 53209: OK
21846 + 27060 = 48906 == 48906: OK
76 * 57 = 4332 == 4332: OK
24213 != 23693 = True == True: OK
21894 / 16710 = 1 == 1: OK
15592 / 12470 = 1 == 1: OK
10574 % 8974 = 1600 == 1600: OK
31981 XOR 27615 = 5938 == 5938: OK
```

### Multiplication-only training

Wenn training only operation multiplication `*` by setting `valid_ops=['*']`, success after about 100 epochs can be
observed with 12-layer dense network (1024 neurons) with additive residual-connections and batchnorm between each layer:
```
27250 * 14588 = 397523000 == 397523000: OK
10510 * 17232 = 181108320 == 181108320: OK
25989 * 8009 = 208145901 == 208145901: OK
31096 * 8833 = 274670968 == 274670968: OK
7540 * 31608 = 238324320 == 238324320: OK
20906 * 11601 = 242530506 == 242530506: OK
25014 * 14725 = 368331150 == 368331150: OK
18460 * 31792 = 586880320 != 721098048: Error
0b101010111110110001010101000000
0b100010111110110001010101000000
OP*: Ok: 82, Error: 18 -> 82.0%
```
'Bitflips' in case of error simply became much less frequent after the network size reached a certain depth (12 layers, residual) and size (1024).

## Different hardware platforms

Currently support are

- Local jupyter instances with a local graphics card
- Mac M1 metal graphics with local jupyter instance and Apple's [tensorflow-plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- Google Colab instances with either GPU or TPU runtime. The colab version uses a Google Drive account to cache data and model state within a Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.

It's possible to start training on one platform (e.g. Google Colab TPU) and then transfer the result to local hardware (e.g. a Mac M1).

Colab notebooks store the trained model in the user's Google Drive (the notebook will request access to the drive) at `My Drive/Colab Notebooks/ALU_Net`.
For local jupyter instances both training data caches and model are stored in the directory of the notebook.

All hardware (Colab GPUs) and local GPUs, with exception of Colab TPUs store the complete model state in a directory `math_model_*` that contains are parameter (weights, optimizer states) or the training process. Simply copy that directory to tranfer the training progress.

Since (as far as I know) exporting the complete model for TPUs to local colab (or drive) instances is not supported, TPUs only save the model weights as single file `math_model_*.h5`. It's possible to copy that file from Google Drive to a local jupyter instance and it will be imported on the first run. Thus training from TPUs can be transfered to local machines.

### Performance

Model 4x512 residual

* Google TPU v2: 29ms / step    (Dataset probably too small for efficient parallelisation)
* Nvidia 1080ti: 45ms / step
* Tesla P100: 57ms / step
* Apple M1: 219ms / step

## Customizing models and training

* The function create_load_model() creates (or loads the state from either Google Drive or local filesystem) one several pre-defined models: e.g. `minimal`, `medium` and `large`, referred in dictionary `model_variants`. To try different models simply at one to this function. At training-time the model is selected by the global `model_variant=`
* The global `valid_ops` determines if all `op_count` (default: 12) ALU operation (`+`, `-`, `*`, .. `!=`) are used or only a subset. If `valid_ops` is `None`, all operations are trained, alternatively a list of operations can be given: `valid_ops = ['*', '/']` would only train the model on multiplication and division.

## Unsorted notes

### Notes on experiments & History

- (2022-03-12) Dataset and boiler-plate code port to [ml-indie-tools](https://github.com/domschl/ALU_Net) completed. Some benchmarks showing same code
  being used by Apple M1, Nvidia and TPU via `ml-indie-tools`.
- (2022-01-06) Python files abstracted and moved to [ml-indie-tools](https://github.com/domschl/ml-indie-tools) [WIP, ongoing]
- (2021-12-09) Positional encoding is crucial for self-attention, since it doesn't know about order. Each of the three word-vectors for op1, operation, and op2 has now three bits added, '1 0 0' for op1, '0 1 0' for operation and '0 0 1' for op2.
- (2021-12-06) 'word-vector-mode' added for use with self-attention layers.
- (2021-12-05) ALU_Dataset has been generalized to arbitrary `bit_counts`. It seems that a 16 bit ALU does successfully train all operations. Increasing `bit_count`, works more or less effortless for all operations other than multiplication.
- (2021-12-01) Multiplication seems to be another case of working once more data and compute is thrown at the problem. 12 dense layers with 1024 neurons and additive residuals between each layer achieves 87% ok after about 100 epochs...
- (2021-11-30) Most difficult operation for the net to learn is always `*`. All bitwise operations and comparisations are usually learned quickly with almost any architecture, followed by `+`, `-` and interestingly also `/` and even `%`. Why this experiment has so much more difficulties in learning how to multiply is currently unknown. Interestingly, if ALU operations are restricted to just `*` (by setting `valid_ops=['*']`), again almost all network architectures start to learn multiplication within a short timespan at least to some degree. Why this doesn't work in the same way with all operations enabled is currently unknown. 

### Colab and remote Tensorboard via Google Drive sync

If you have Google Drive sync for the directory `My Drive/Colab Notebooks/ALU_Net/logs` activated on a remote desktop, a remote Tensorboard instance can display the training progress. You simply need to figure out the physical destination of the files Google Drive syncs. On a Mac that would be by default like this:

```
tensorboard --logdir /Volumes/GoogleDrive/My\ Drive/Colab\ Notebooks/ALU_Net/logs
```

__Note:__ This option is now _disabled by default_ (`MLEnv.init_paths(log_to_gdrive=False)`), because Google copy performance between colab and drive is catastrophicaly slow.

### TPU Notes

- *Saving state*: Saving TPU with Colab notebooks is a challenge. Saving complete models (via `keras.Model.save`)is not supported just with Colab & TPU. It is possible to export the weights (`keras.Model.get_weights`), import them into a model in CPU-scope and save there: `cpu_model.set_weights(tpu_model.get_weights()); cpu_model.save_weights(weights_filename)`. So far I couldn't figure out a similar method for preserving optimizer state.
- *TPU scheduling*: TPU experiments on Colab require frequent restarts, since while the actual TPU transaction remains fast, the runtime slows down geometrically. During first run, TPU is 10x faster than a medium GPU, but this advantage decays within a few hours.
