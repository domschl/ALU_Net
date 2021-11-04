# ALU_Net [WIP]

<a href="https://colab.research.google.com/github/domschl/ALU_Net/blob/main/ALU_Net.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

Create an ALU (arithmetic logic unit) via neural nets using Colab TPUs, GPUs are on local Jupyter instances.
If used with Colab, user's Google Drive is used for persistant storage.

It can be used with Mac M1 GPU, if [Apple's tensorflow plugin](https://developer.apple.com/metal/tensorflow-plugin/) is installed.

If used with Google Colab, a Google Drive account is used to save trained model data and tensorboard logs to your google drive.
This allows easy exporting of trained nets and using remote Tensorboard instances to monitor the training on Colab.

_This is work in progress_

## Experiment

<img align="right" width="300" src="https://github.com/domschl/ALU_Net/blob/main/resources/ALU.png">

The neural network is expected to learn arithmetic and logic operations between two unsigned 15 bit integers. The possible operations are: `+`, `-`, `/` (integer division), `*`, `%` (modulo), `AND` boolean bitwise AND, `OR` boolean bitwise OR, '`XOR` boolean bitwise XOR and the comparators `=`, `<`, `>`, `!=`.
Each integer is encoded as 16 input neurons [only 15 bits are currently used, positive ints only -- this might change] (`0.0` level for bit zero, `1.0` level for bit one), the operation is encoded binary as 4 neurons (e.g. addition: `0.0, 0.0, 0.0, 0.0`; subtraction: `0.0, 0.0, 0.0, 1.0` etc.)
The result of the network is a 32 bit integer again binary encoded. The value `True` is encoded as `0b1111111111111111`, `False` is `0b0000000000000000`.
The input vector has dimension 36 (`2*16 + 4`), the output has dimension 32.

### Example results after a short period of training

This shows lines in format:


```
<int> <operation> <int> = <net-eval> ==/!= <actual result> OK/Error
```

In case of an error, a binary representation of expected and actual result are shown.
The most difficult operation for the net to learn seems to be multiplication.
```
31720 / 4697 = 6 == 6: OK
31548 > 19449 = True == True: OK
22830 / 10708 = 2 == 2: OK
22706 > 12157 = True == True: OK
30303 XOR 16698 = 14181 == 14181: OK
21708 - 15004 = 6704 == 6704: OK
27889 + 24939 = 52828 == 52828: OK
14648 - 13524 = 1124 == 1124: OK
24008 > 3930 = True == True: OK
1738 OR 9412 = 9934 == 9934: OK
23242 * 32365 = 752227330 != 752004498: Error
0b101100110100101010110110010010
0b101100110101100001010000000010
22119 - 19975 = 2144 == 2144: OK
22712 - 15976 = 6736 == 6736: OK
31518 < 107 = False == False: OK
27517 = 10434 = False == False: OK
26401 - 17991 = 8410 == 8410: OK
4461 = 4461 = True == True: OK
29563 OR 14112 = 30587 == 30587: OK
27995 < 31100 = True == True: OK
30722 XOR 4987 = 27513 == 27513: OK
17986 AND 15620 = 1024 == 1024: OK
20433 % 17561 = 2872 == 2872: OK
```

## Different hardware platforms

Currently support are

- Local jupyter instances with a local graphics card
- Mac M1 metal graphics with local jupyter instance and Apple's tensorflow-plugin
- Google Colab instances with either GPU or TPU runtime. The colab version uses a Google Drive account to cache data and model state within a Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.

It's possible to start training on one platform (e.g. Google Colab TPU) and then transfer the result to local hardware (e.g. a Mac M1).

Colab notebooks store the trained model in the user's Google Drive (the notebook will request access to the drive) at `My Drive/Colab Notebooks/ALU_Net`.
For local jupyter instances both training data caches and model are stored in the directory of the notebook.

All hardware (Colab GPUs) and local GPUs, with exception of Colab TPUs store the complete model state in a directory `math_model_*` that contains are parameter (weights, optimizer states) or the training process. Simply copy that directory to tranfer the training progress.

Since (as far as I know) exporting the complete model for TPUs to local colab (or drive) instances is not supported, TPUs only save the model weights as single file `math_model_*.h5`. It's possible to copy that file from Google Drive to a local jupyter instance and it will be imported on the first run. Thus training from TPUs can be transfered to local machines.

## Customizing models and training

* The function create_load_model() creates (or loads the state from either Google Drive or local filesystem) one of three currently defined models: `minimal`, `medium` and `large`. To try different models simply at one to this function. At training-time the model is selected by the global `model_type=`
* The global `valid_ops` determines if all 12 ALU operation (`+`, `-`, `*`, .. `!=`) are used or only a subset. If `valid_ops` is `None`, all 12 operations are trained, alternatively a list of operations can be given: `valid_ops = ['*', '/']` would only train the model on multiplication and division.

## Unsorted notes

### TPU Notes

- *Saving state*: Saving TPU with Colab notebooks is a challenge. Saving complete models (via `keras.Model.save`)is not supported just with Colab & TPU. It is possible to export the weights (`keras.Model.get_weights`), import them into a model in CPU-scope and save there: `cpu_model.set_weights(tpu_model.get_weights()); cpu_model.save_weights(weights_filename)`. So far I couldn't figure out a similar method for preserving optimizer state.
- *TPU scheduling*: TPU experiments on Colab require frequent restarts, since while the actual TPU transaction remains fast, the runtime slows down geometrically. During first run, TPU is 10x faster than a medium GPU, but this advantage decays within a few hours.
