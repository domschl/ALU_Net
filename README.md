# ALU_Net [WIP]

<a href="https://colab.research.google.com/github/domschl/ALU_Net/blob/main/ALU_Net.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

Create an ALU (arithmetic logic unit) via neural nets using Colab TPUs, GPUs are on local Jupyter instances.
If used with Colab, user's Google Drive is used for persistant storage.

It can be used with Mac M1 GPU, if [Apple's tensorflow plugin](https://developer.apple.com/metal/tensorflow-plugin/) is installed.

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

## Unsorted notes

### TPU Notes

- *Saving state*: Saving TPU with Colab notebooks is a challenge. Saving complete models (via `keras.Model.save`)is not supported just with Colab & TPU. It is possible to export the weights (`keras.Model.get_weights`), import them into a model in CPU-scope and save there: `cpu_model.set_weights(tpu_model.get_weights()); cpu_model.save_weights(weights_filename)`. So far I couldn't figure out a similar method for preserving optimizer state.
- *TPU scheduling*: TPU experiments on Colab require frequent restarts, since while the actual TPU transaction remains fast, the runtime slows down geometrically. During first run, TPU is 10x faster than a medium GPU, but this advantage decays within a few hours.
