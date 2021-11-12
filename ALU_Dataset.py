
import sys
import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

from ml_env_tools import MLEnv

# ## Training data
class ALU_Dataset():
    """ Generate training data for all ALU operations """
    # The ALU takes two integers and applies one of the supported
    # model_ops. Eg op1=123, op2=100, op='-' -> result 23
    # The net is supposed to learn to 'calculate' the results for
    # arbitrary op1, op2 (positive integers, 0..32767) and 
    # the twelve supported ops 

    def __init__(self, ml_env:MLEnv, pre_weight=False):
        self.ml_env=ml_env
        self.model_ops = ["+", "-", "*", "/", "%",
                          "AND", "OR", "XOR", ">", "<", "=", "!="]
        self.model_is_boolean = [False, False, False, False, False,
                                 False, False, False, True, True, True, True]
        # Probabilites for creating a sample for each of the ops, (Will be
        # reweighted on checks to generate for samples for 'difficult' ops):
        self.model_dis = [10, 10, 10, 10, 10, 10,   10,  10,   10, 10, 10, 10]
        model_dis_w = [19, 12, 110, 15, 36, 10, 10, 10, 10, 10, 10, 10]
        self.model_funcs = [self.add_smpl, self.diff_smpl, self.mult_smpl,
                            self.div_smpl, self.mod_smpl, self.and_smpl,
                            self.bor_smpl, self.xor_smpl, self.greater_smpl,
                            self.lesser_smpl, self.eq_smpl, self.neq_smpl]
        self.bit_count = 15
        self.all_bits_one = 0x7fffffff
        self.true_vect = self.all_bits_one
        self.false_vect = 0
        if pre_weight is True:
            self.model_dis=model_dis_w

    @staticmethod
    def int_to_binary_vect(num_int, num_bits=8):
        """ get a binary encoded vector of n of bit-lenght nm """
        num_vect = np.zeros(num_bits, dtype=np.float32)
        for i in range(0, num_bits):
            if num_int & (2**i) != 0:
                num_vect[i] = 1.0
        return num_vect

    @staticmethod
    def int_to_onehot_vect(num_int, num_bits=12):
        """ get a one-hot encoded vector of n of bit-lenght nm """
        num_vect = np.zeros(num_bits, dtype=np.float32)
        num_vect[num_int] = 1.0
        return num_vect

    @staticmethod
    def get_random_bits(bits):
        """ get bits random int 0...2**bits-1 """
        return random.randint(0, 2**bits-1)

    def op_string_to_index(self, op_string):
        """ transform op_string (e.g. '+' -> 0) into corresponding index """
        for i in range(0, len(self.model_ops)):
            if self.model_ops[i] == op_string:
                return i
        return -1

    def get_data_point(self, equal_distrib=False, short_math=False, valid_ops=None):
        """ Get a random example for on ALU operation for training """
        result = -1
        op1 = self.get_random_bits(self.bit_count)
        op2 = self.get_random_bits(self.bit_count)
        if valid_ops is not None and len(valid_ops)==0:
            valid_ops=None
        if valid_ops is not None:
            if equal_distrib is False:
                print("Op restriction via valid_ops forces equal_distrib=True")
                equal_distrib=True
            for op in valid_ops:
                if op not in self.model_ops:
                    print(f'Cannot restrict valid_ops to {op}, unknown operation, ignoring all valid_ops')
                    valid_ops=None
                    break

        if equal_distrib or valid_ops is not None:
            if valid_ops is None:   
                op_index = random.randint(0, len(self.model_ops)-1)
            else:
                if len(valid_ops)==1:
                    op_index=0
                else:
                    op_index = random.randint(0, len(valid_ops)-1)
                op_index=self.model_ops.index(valid_ops[op_index])
        else: # make 'difficult' ops more present in training samples:
            rx = 0
            for md in self.model_dis:
                rx += md
            rrx = random.randint(0, rx)
            rx = 0
            op_index = 0
            for op_index in range(0, len(self.model_ops)):
                rx += self.model_dis[op_index]
                if rx > rrx:
                    break
        return self.encode_op(op1, op2, op_index, short_math)

    def generator(self, samples=20000, equal_distrib=False, short_math=False, valid_ops=None):
        while True:
            x, Y = self.create_training_data(samples=samples, short_math=short_math, valid_ops=valid_ops, verbose=False, title=None)
            #x, Y, _, _, _ = self.get_data_point(equal_distrib=equal_distrib, short_math=short_math, valid_ops=valid_ops)
            yield x, Y

    def encode_op(self, op1, op2, op_index, short_math=False):
        """ turn two ints and operation into training data """
        op1, op2, result = self.model_funcs[op_index](op1, op2, short_math)
        if self.model_is_boolean[op_index] is True:
            if result==self.false_vect:
                str_result="False"
            elif result==self.true_vect:
                str_result="True"
            else:
                str_result="undefined"
        else:
            str_result=result
        sym = f"{op1} {self.model_ops[op_index]} {op2} = {str_result}"
        inp = np.concatenate(
            [self.int_to_binary_vect(op1, num_bits=16),
            self.int_to_onehot_vect(op_index, num_bits=12),
            self.int_to_binary_vect(op2, num_bits=16)])

        oup = self.int_to_binary_vect(result, num_bits=32)
        return inp, oup, result, op_index, sym

    @staticmethod
    def add_smpl(op1, op2, _):
        """ addition training example """
        result = op1+op2
        return op1, op2, result

    @staticmethod
    def diff_smpl(op1, op2, _):
        """ subtraction training example """
        if op2 > op1:
            op2, op1 = op1, op2
        result = op1-op2
        return op1, op2, result

    @staticmethod
    def mult_smpl(op1, op2, short_math=False):
        """ multiplication training example """
        if short_math:
            op1 = op1 % 1000
            op2 = op2 % 1000
        result = op1*op2
        return op1, op2, result

    def div_smpl(self, op1, op2, _):
        """ integer division training example """
        while op2 == 0:
            op2 = self.get_random_bits(self.bit_count)
        if op1 < op2 and random.randint(0, 2) != 0:
            if op1 != 0:
                op1, op2 = op2, op1
        result = op1//op2
        return op1, op2, result

    def mod_smpl(self, op1, op2, _):
        """ modulo (remainder) training example """
        while op2 == 0:
            op2 = self.get_random_bits(self.bit_count)
        if op1 < op2 and random.randint(0, 2) != 0:
            if op1 != 0:
                op1, op2 = op2, op1
        result = op1 % op2
        return op1, op2, result

    @staticmethod
    def and_smpl(op1, op2, _):
        """ bitwise AND training example """
        result = op1 & op2
        return op1, op2, result

    @staticmethod
    def bor_smpl(op1, op2, _):
        """ bitwise OR training example """
        result = op1 | op2
        return op1, op2, result

    @staticmethod
    def xor_smpl(op1, op2, _):
        """ bitwise XOR training example """
        result = op1 ^ op2
        return op1, op2, result

    def greater_smpl(self, op1, op2, _):
        """ integer comparisation > training example """
        if op1 > op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def lesser_smpl(self, op1, op2, _):
        """ integer comparisation < training example """
        if op1 < op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def eq_smpl(self, op1, op2, _):
        """ integer comparisation == training example """
        if random.randint(0, 1) == 0:  # create more cases
            op2 = op1
        if op1 == op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def neq_smpl(self, op1, op2, _):
        """ integer comparisation != training example """
        if random.randint(0, 1) == 0:  # create more cases
            op2 = op1
        if op1 != op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def create_data_point(self, op1, op2, op_string):
        """ create training data from given ints op1, op2 and op_string """
        op_index = self.op_string_to_index(op_string)
        if op_index == -1:
            print(f"Invalid operation {op_string}")
            return np.array([]), np.array([]), -1, -1, None
        return self.encode_op(op1, op2, op_index)

    def create_training_data(self, samples=10000, short_math=False, valid_ops=None, verbose=True, title=None):
        """ create a number of training samples """
        x, y, _, _, _ = self.get_data_point()
        dpx = np.zeros((samples, len(x)), dtype=np.float32)
        dpy = np.zeros((samples, len(y)), dtype=np.float32)
        if verbose is True:
            if title is None:
                print(f"Creating {samples} data points (. = 1000 progress)")
            else:
                print(f"{title}: Creating {samples} data points (. = 1000 progress)")

        for i in range(0, samples):
            if verbose is True:
                if i%100000 == 0:
                    print(f"{i:>10} ", end="")
            if (i+1) % 1000 == 0:
                if verbose is True:
                    print(".", end="")
                    sys.stdout.flush()
                    if (i+1) % 100000 == 0:
                        print()
            if valid_ops is None:
                x, y, _, _, _ = self.get_data_point(
                    equal_distrib=False, short_math=short_math)
            else:
                x, y, _, _, _ = self.get_data_point(
                    equal_distrib=True, short_math=short_math, valid_ops=valid_ops)
            dpx[i, :] = x
            dpy[i, :] = y
        if verbose is True:
            print()
        return dpx, dpy

    def create_dataset(self, samples=10000, batch_size=2000, short_math=False, valid_ops=None, name=None, cache_path=None, use_cache=True, regenerate_cached_data=False):
        is_loaded=False
        if use_cache is True and cache_path is None:
            print("can't use cache if no cache_path is given, disabling cache!")
            use_cache = False
        if use_cache is True:
            if valid_ops is not None:
                infix='_'
                for vo in valid_ops:
                    if vo=='*': # Prevent poison-filenames
                        vo="MULT"
                    if vo=='/':
                        vo="DIV"
                    if vo=='%':
                        vo='MOD'
                    if vo=='<':
                        vo='LT'
                    if vo=='>':
                        vo='GT'
                    if vo=='=':
                        vo='EQ'
                    if vo=='!=':
                        vo='NE'
                    infix+=vo
            else:
                infix=""
            cache_file_x=os.path.join(cache_path, f"{name}_{infix}_{samples}_x.npy")
            cache_file_Y=os.path.join(cache_path, f"{name}_{infix}_{samples}_Y.npy")
        if use_cache is True  and regenerate_cached_data is False and os.path.exists(cache_file_x) and os.path.exists(cache_file_Y):
            try:
                x = np.load(cache_file_x, allow_pickle=True)
                Y = np.load(cache_file_Y, allow_pickle=True)
                if len(x)==samples:
                    is_loaded=True
                    print(f"Data {name} loaded from cache")
                else:
                    print(f"Sample count has changed from {len(x)} to {samples}, regenerating {name} data...")
            except Exception as e:
                print(f"Something went wrong when loading {cache_file_x}, {cache_file_Y}: {e}")
        if is_loaded is False:
            x, Y = self.create_training_data(samples=samples, short_math=short_math, valid_ops=valid_ops, title=name)
            if use_cache is True:
                print(f"Writing data-cache {cache_file_x}, {cache_file_Y}...", end="")
                np.save(cache_file_x, x, allow_pickle=True)
                print(", x", end="")
                np.save(cache_file_Y, Y, allow_pickle=True)
                print(", Y, done.")
        shuffle_buffer=10000
        dataset=tf.data.Dataset.from_tensor_slices((x, Y)).cache()
        dataset=dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        if self.ml_env.is_tpu is True:
            dataset=dataset.repeat() # Mandatory for Keras TPU for now
        dataset=dataset.batch(batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
        dataset=dataset.prefetch(-1) # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
        return dataset

    def create_dataset_from_generator(self, short_math=False, valid_ops=None):
        dataset=tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                    tf.TensorSpec(shape=(None,44), dtype=np.float32),
                    tf.TensorSpec(shape=(None,32), dtype=np.float32))
            )
        return dataset

    def get_datasets(self, pre_weight=True, samples=100000, validation_samples=10000, batch_size=2000, short_math=False, valid_ops=None, cache_path=None, use_cache=True, regenerate_cached_data=False):
        train = self.create_dataset(samples=samples, batch_size=batch_size, short_math=short_math, valid_ops=valid_ops,
                                        name="train",cache_path=cache_path, use_cache=use_cache, regenerate_cached_data=regenerate_cached_data)
        val = self.create_dataset(samples=validation_samples, batch_size=batch_size, short_math=short_math, valid_ops=valid_ops,
                                    name="validation",cache_path=cache_path, use_cache=use_cache, regenerate_cached_data=regenerate_cached_data)
        return train, val

    @staticmethod
    def decode_results(result_int_vects):
        """ take an array of 32-float results from neural net and convert to ints """
        result_vect_ints = []
        for vect in result_int_vects:
            if (len(vect) != 32):
                print(f"Ignoring unexpected vector of length {len(vect)}")
            else:
                int_result = 0
                for i in range(0, 32):
                    if vect[i] > 0.5:
                        int_result += 2**i
                result_vect_ints.append(int_result)
        return result_vect_ints

    def check_results(self, model, samples=1000, short_math=False, valid_ops=None, verbose=False):
        """ Run a number of tests on trained model """
        ok = 0
        err = 0
        operr = [0]*len(self.model_ops)
        opok = [0]*len(self.model_ops)
        for _ in range(0, samples):
            x, _, z, op, s = self.get_data_point(
                equal_distrib=True, valid_ops=valid_ops, short_math=short_math)
            res = self.decode_results(model.predict(np.array([x])))
            if res[0] == z:
                ok += 1
                opok[op] += 1
                r = "OK"
            else:
                err += 1
                operr[op] += 1
                r = "Error"
            if verbose is True:
                if self.model_is_boolean[op] is True:
                    if res[0]==self.false_vect:
                        str_result="False"
                    elif res[0]==self.true_vect:
                        str_result="True"
                    else:
                        str_result="undefined"
                else:
                    str_result=res[0]
                if res[0]==z:
                    print(f"{s} == {str_result}: {r}")
                else:
                    print(f"{s} != {str_result}: {r}")
                    if self.model_is_boolean[op] is False:
                        print(bin(res[0]))
                        print(bin(z))
        opsum = ok+err
        if opsum == 0:
            opsum = 1
        print(f"Ok: {ok}, Error: {err} -> {ok/opsum*100.0}%")
        print("")
        for i in range(0, len(self.model_ops)):
            opsum = opok[i]+operr[i]
            if opsum == 0:
                continue
            # modify the distribution of training-data generated to favour
            # ops with bad test results, so that more training data is
            # generated on difficult cases:
            self.model_dis[i] = int(operr[i]/opsum*100)+10
            print(
                f"OP{self.model_ops[i]}: Ok: {opok[i]}, Error: {operr[i]}", end="")
            print(f" -> {opok[i]/opsum*100.0}%")
        if valid_ops == None:
            print("Change probability for ops in new training data:")
            print(f"Ops:     {self.model_ops}")
            print(f"Weights: {self.model_dis}")

