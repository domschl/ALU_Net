# ALU_Net [WIP]
Create an ALU (arithmetic logic unit) via neural nets using Colab TPUs

_This is work in progress_

## Experiment

<img align="right" width="300" src="https://github.com/domschl/ALU_Net/blob/main/resources/ALU.png">
The neural network is expected to learn arithmetic and logic operations between two unsigned 15 bit integers. The possible operations are: '`+`', '`-`', '`/`' (integer division), '`*`', '`%`' (modulo), '`AND`' boolean bitwise and, '`OR`' boolean bitwise or, '`XOR`' boolean bitwise xor and the comparators '`=`', '`<`', '`>`', '`!=`'.
Each integer is encoded as 16 input neurons [only 15 bits are currently used, positive ints only -- this might change] (`0.0` level for bit zero, `1.0` level for bit one), the operation is encoded binary as 4 neurons (e.g. addition: `0.0, 0.0, 0.0, 0.0`; subtraction: `0.0, 0.0, 0.0, 1.0` etc.)
The result of the network is a 32 bit integer again binary encoded. The value `True` is encoded as `0b1111111111111111`, `False` is `0b0000000000000000`.
The input vector has dimension 36 (`2*16 + 4`), the output has dimension 32.

### Example results after a short period of training

This shows one the left two integers with on of the 12 operations, followed by '`=`' and the result the net determined,
followed by '`==`' and the actual correct result. In case of an error, a binary representation of expected and actual 
result are shown.
The most difficult operation for the net to learn seems to be multiplication.
```
12529+29504=42033 == 42033: OK
9623*30041=289084543 != 289304175: Error
0b10001001111100110111001101111
0b10001001110110001010001111111
26068!=26068=false == false: OK
14305AND456=448 == 448: OK
25739/19884=1 == 1: OK
19257%17615=1642 == 1642: OK
26890!=1089=true == true: OK
29480-2900=26580 == 26580: OK
3904AND6485=2368 == 2368: OK
11445XOR17665=27060 == 27060: OK
30464=30464=true == true: OK
14274%4064=2082 == 2082: OK
6990<26236=true == true: OK
30026%7531=7433 == 7433: OK
32160%25702=6458 == 6458: OK
8307!=28150=true == true: OK
3899AND12095=3899 == 3899: OK
19943=12772=false == false: OK
23248/21305=1 == 1: OK
8441%399=62 != 210: Error
0b11010010
0b111110
15489XOR7435=8586 == 8586: OK
6249-1987=4262 == 4262: OK
```

## Unsorted notes

### TPU Notes

- *Saving state*: Saving TPU with Colab notebooks is a challenge. Saving complete models (via `keras.Model.save`)is not supported just with Colab & TPU. It is possible to export the weights (`keras.Model.get_weights`), import them into a model in CPU-scope and save there: `cpu_model.set_weights(tpu_model.get_weights()); cpu_model.save_weights(weights_filename)`. So far I couldn't figure out a similar method for preserving optimizer state.
- *TPU scheduling*: TPU experiments on Colab require frequent restarts, since while the actual TPU transaction remains fast, the runtime slows down geometrically. During first run, TPU is 10x faster than a medium GPU, but this advantage decays within a few hours.
