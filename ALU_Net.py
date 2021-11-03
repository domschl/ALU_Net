#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/domschl/ALU_Net/blob/main/ALU_Net.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # A neural net that tries to become an ALU (arithmetic logic unit)
# 
# This notebook can run
# 
# - on local jupyter instances with a local graphics card
# - on Mac M1 with local jupyter instance and [Apple's tensorflow-plugin](https://developer.apple.com/metal/tensorflow-plugin/)
# - on Google Colab instances with either GPU or TPU runtime. The colab version uses a Google Drive account to cache data and model state within a Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.

# ## 1. Configuration and setup

import sys
import os
import random
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

use_keras_project_versions=False
# Namespaces, namespaces
if use_keras_project_versions is False:
    # print("Importing Keras from tensorflow project (it won't work otherwise with TPU)")
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks, metrics, optimizers
else:
    # print("Importing Keras from keras project (which had recently declared independence [again]) -- as recommended")
    import keras
    from keras import layers, regularizers, callbacks, metrics, optimizers


class MLEnv():
    def __init__(self):
        self.is_colab = self.check_colab()
        self.check_hardware()

    @staticmethod
    def check_colab():
        try: # Colab instance?
            from google.colab import drive
            is_colab = True
            get_ipython().run_line_magic('load_ext', 'tensorboard')

            try:
                get_ipython().run_line_magic('tensorflow_version', '2.x')
            except:
                pass
        except: # Not? ignore.
            is_colab = False
            pass
        return is_colab

    # Hardware check:
    def check_hardware(self, verbose=True):
        self.is_tpu = False
        self.is_gpu = False
        self.tpu_address = None

        for hw in ["CPU", "GPU", "TPU"]:
            hw_list=tf.config.experimental.list_physical_devices(hw)
            if len(hw_list)>0:
                if hw=='TPU':
                    self.is_tpu=True
                if hw=='GPU':
                    self.is_gpu=True
                if verbose is True:
                    print(f"{hw}: {hw_list} {tf.config.experimental.get_device_details(hw_list[0])}") 

        if self.is_colab:
            if self.is_tpu is True:
                if self.tpu_address is None:
                    try:
                        self.tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
                        tf.config.experimental_connect_to_host(self.tpu_address)
                        if verbose is True:
                            print(f"TPU available at {self.tpu_address}")
                    except Exception as e:
                        if verbose is True:
                            print("No TPU available: {e}")
                        self.is_tpu = False
                else:
                    if verbose is True:
                        print(f"TPU available, already connected to {self.tpu_address}")

        if not self.is_tpu:
            if not self.is_gpu:
                if verbose is True:
                    print("WARNING: You have neither TPU nor GPU, this is going to be very slow!")
            else:
                if verbose is True:
                    print("GPU available")
        else:
            tf.compat.v1.disable_eager_execution()
            if verbose is True:
                print("TPU: eager execution disabled!")

        def mount_gdrive(self, mount_point="/content/drive", root_path="/content/drive/My Drive", verbose=True):
            if self.is_colab is True:
                if verbose is True:
                    print("You will now be asked to authenticate Google Drive access in order to store training data (cache) and model state.")
                    print("Changes will only happen within Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.")
                if not os.path.exists(root_path):
                    # drive.flush_and_unmount()
                    drive.mount(mount_point) #, force_remount=True)
                    return True, root_path
                if not os.path.exists(root_path):
                    print(f"Something went wrong with Google Drive access. Cannot save model to {root_path}")
                    return False, None
                else:
                    return True, root_path
            else:
                if verbose is True:
                    print("You are not on a Colab instance, so no Google Drive access is possible.")
                return False, None

        def init_paths(self, project_name='project', model_name='model', model_variant=None, log_to_gdrive=False):
            self.save_model = True
            self.model_file=None
            self.cache_stub=None
            self.weights_file = None
            self.project_path = None
            self.log_path = "./logs"
            self.log_mirror_path = None
            if self.is_colab:
                self.save_model, self.root_path = mount_gdrive()
            else:
                self.root_path='.'

            if self.save_model:
                if self.is_colab:
                    self.project_path=os.path.join(self.root_path,f"Colab Notebooks/{project_name}")
                    if log_to_gdrive is True:
                        self.log_mirror_path = os.path.join(self.root_path,f"Colab Notebooks/{project_name}/logs")
                else:
                    self.project_path=self.root_path
                if model_variant is None:
                    self.model_file=os.path.join(self.project_path,f"{model_name}.h5")
                    self.weights_file=os.path.join(self.project_path,f"{model_name}_weights.h5")
                else:
                    self.model_file=os.path.join(self.project_path,f"{model_name}_{model_variant}.h5")
                    self.weights_file=os.path.join(self.project_path,f"{model_name}_{model_variant}_weights.h5")
                self.cache_stub=os.path.join(self.project_path,'data_cache')
                if self.is_tpu is False:
                    print(f"Model save-path: {self.model_file}")
                else:
                    print(f"Weights save-path: {self.weights_file}")
                print(f'Data cache file-stub {self.cache_stub}')
            return self.model_file, self.weights_file, self.cache_stub, self.log_path


# ## Training data
class ALU_Dataset():
    """ Generate training data for all ALU operations """
    # The ALU takes two integers and applies one of the supported
    # model_ops. Eg op1=123, op2=100, op='-' -> result 23
    # The net is supposed to learn to 'calculate' the results for
    # arbitrary op1, op2 (positive integers, 0..32767) and 
    # the twelve supported ops 

    def __init__(self, pre_weight=False):
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
             self.int_to_binary_vect(op_index, num_bits=4),
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

    def create_dataset(self, samples=10000, batch_size=2000, short_math=False, valid_ops=None, name=None, cache_file=None, use_cache=True, regenerate_cached_data=False):
        is_loaded=False
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
            cache_file_x=cache_file+infix+'_x.npy'
            cache_file_Y=cache_file+infix+"_Y.npy"
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
                print(f"Writing data-cache {cache_file_x}, {cache_file_Y}...")
                np.save(cache_file_x, x, allow_pickle=True)
                np.save(cache_file_Y, Y, allow_pickle=True)
        shuffle_buffer=10000
        dataset=tf.data.Dataset.from_tensor_slices((x, Y)).cache()
        dataset=dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        if is_tpu is True:
            dataset=dataset.repeat() # Mandatory for Keras TPU for now
        dataset=dataset.batch(batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
        dataset=dataset.prefetch(-1) # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
        return dataset

    def create_dataset_from_generator(self, short_math=False, valid_ops=None):
        dataset=tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                    tf.TensorSpec(shape=(None,36), dtype=np.float32),
                    tf.TensorSpec(shape=(None,32), dtype=np.float32))
            )
        return dataset
        
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


# In[ ]:


def get_datasets(pre_weight=True, samples=100000, validation_samples=10000, batch_size=2000, short_math=False, valid_ops=None, cache_file_stub='cache', use_cache=True, regenerate_cached_data=False):
    math_data = ALU_Dataset(pre_weight=pre_weight)
    train = math_data.create_dataset(samples=samples, batch_size=batch_size, short_math=short_math, valid_ops=valid_ops,
                                     name="training-data",cache_file=cache_file_stub+"_train", use_cache=use_cache, regenerate_cached_data=regenerate_cached_data)
    val = math_data.create_dataset(samples=validation_samples, batch_size=batch_size, short_math=short_math, valid_ops=valid_ops,
                                   name="validation-data",cache_file=cache_file_stub+"_val", use_cache=use_cache, regenerate_cached_data=regenerate_cached_data)
    return math_data, train, val


# ## Different models

# In[ ]:


def model_large(inputs, regu1=1e-7, regu2=1e-7, regu3=1e-7, neurons=1024, filters=16, strides=2, kernel_size=3):  # neurons must be divisible by 4 for the residual below
    # Input goes parallel into 3 streams which will be combined at the end:
    # Stream 1: convolutions
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)  # x0)
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x1 = d1(rinp)
    filters=filters*2
    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x2 = d2(x1)
    filters=filters*2
    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x3 = d3(x2)
    filters=filters*2
    d4 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xcvl = d4(x3)
    flatter = layers.Flatten()
    xf = flatter(xcvl)
    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xe1 = de1(xf)

    # Stream 2: simple dense layers
    df1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf1 = df1(inputs)
    df2 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf2 = df2(xf1)
    df3 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf3 = df3(xf2)

    # Stream3: dense layers with residual pathway
    dfa1 = layers.Dense(neurons/4, kernel_regularizer=regularizers.l2(
        regu3), activation="relu")
    xfa1 = dfa1(inputs)
    dfa2 = layers.Dense(neurons/4, kernel_regularizer=regularizers.l2(
        regu3), activation="relu")
    con0 = layers.Concatenate()
    xfa2 = con0([dfa2(xfa1),xfa1])
    dfa3 = layers.Dense(neurons/2, kernel_regularizer=regularizers.l2(regu3), activation="relu")
    con1 = layers.Concatenate()
    xfa3 = con1([dfa3(xfa2),xfa2])

    # Concat of stream1,2,3
    con = layers.Concatenate()
    xcon = con([xe1, xf3, xfa3])
    dc1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xc1 = dc1(xcon)

    # Use sigmoid to map to bits 0..1
    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xc1)
    return outputs


# In[ ]:


def model_medium(inputs, regu1=1e-7, regu2=1e-7, neurons=256, lstm_neurons=128, filters=64, kernel_size=3, strides=2):
    #df1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
    #    regu1), activation="relu")
    #xf1 = df1(inputs)

    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs) # xf1)
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x1 = d1(rinp)
    filters = 2*filters
    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x2 = d2(x1)
    filters = 2*filters
    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x3 = d3(x2)

    flatter = layers.Flatten()
    xf = flatter(x3)

    r1 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr1 = r1(rinp)
    r2 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr2 = r2(xr1)
    r3 = layers.LSTM(lstm_neurons)
    xr3 = r3(xr2)

    cc = layers.Concatenate()
    xc = cc([xf, xr3])

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xe1 = de1(xc)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs


# In[ ]:


def model_minimal(inputs, neurons=64, regu1=1e-7):
    df1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf1 = df1(inputs)
    df2 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf2 = df2(xf1)
    df3 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf3 = df3(xf2)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xf3)
    return outputs


# In[ ]:


def model_minimal_recurrent(inputs, neurons=64, lstm_neurons=128, regu1=1e-7):
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)
    r1 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr1 = r1(rinp)
    r2 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr2 = r2(xr1)
    r3 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr3 = r3(xr2)

    flatter = layers.Flatten()
    xf = flatter(xr3)

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xe1 = de1(xf)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs


# In[ ]:


def model_conv1d_recurrent(inputs, neurons=512, lstm_neurons=386, filters=128, kernel_size=3, strides=2, regu0=1e-7, regu1=1e-7, regu2=1e-7):
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)

    r1 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr1 = r1(rinp)
    
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x1 = d1(xr1)
    
    r2 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr2 = r2(x1)

    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(regu2), activation="relu")
    x2 = d2(xr2)

    r3 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr3 = r3(x2)

    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(regu2), activation="relu")
    x3 = d3(xr3)

    flatter = layers.Flatten()
    xf = flatter(x3)

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xe1 = de1(xf)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs


# In[ ]:


model_dict = {"large": model_large,
              "medium": model_medium,
              "minimal": model_minimal,
              "minimal_recurrent": model_minimal_recurrent,
              "conv_recurrent": model_conv1d_recurrent
              }

def create_load_model(save_path=None, model_type='large'):
    """ Create or load a model """
    if save_path is None or not os.path.exists(save_path): #or is_tpu is True:
        print("Initializing new model...")
        inputs = keras.Input(shape=(36,))  # depends on encoding of op-code!
        if model_type not in model_dict:
            print('Unkown model type')
            return None
        outputs = model_dict[model_type](inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name="maths_"+model_type)
        print(f"Compiling new model of type {model_type}")
        if use_keras_project is False: 
            opti = keras.optimizers.Adam(learning_rate=0.001)
        else:
            opti = optimizers.Adam(learning_rate=0.001)
        model.compile(loss="mean_squared_error", optimizer=opti, metrics=[metrics.MeanSquaredError(), 'accuracy'])
    else:
        print(f"Loading standard-format model of type {model_type} from {model_file}")
        model = tf.keras.models.load_model(model_file)
        print("Continuing training from existing model")
    model.summary()
    return model

def get_model(save_path=None, on_tpu=False, model_type='large', import_weights=False):
    if is_tpu is True and on_tpu is True:
        tpu_is_init=False
        if tpu_is_init is False:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)    
            tpu_is_init=True
        with tpu_strategy.scope():
            print("Creating TPU-scope model")
            model=create_load_model(save_path=None, model_type=model_type)
        if weights_file is not None and os.path.exists(weights_file):
            print("Injecting saved weights into TPU model, loading...")
            temp_model = create_load_model(save_path=None, model_type=model_type)
            temp_model.load_weights(weights_file)
            print("Injecting...")
            model.set_weights(temp_model.get_weights())
            print("Updated TPU weights from saved model")
        return model
    else:
        print("Creating standard-scope model")
        model = create_load_model(save_path=save_path, model_type=model_type)
        if import_weights is True and weights_file is not None and os.path.exists(weights_file):
            print("Injecting saved weights into model, loading...")        
            model.load_weights(weights_file)
            imported_weights_file = weights_file+'-imported'
            os.rename(weights_file, imported_weights_file)
            print(f"Renamed weights file {weights_file} to {imported_weights_file} to prevent further imports!")
        return model


# In[ ]:


# This is seriously risky code!
flush_timer = 0
flush_timeout = 180   # don't set this too low, otherwise your local gdrive won't ever sync!
def gdrive_flusher(epoch, logs):
    return
#    global flush_timer
#    global flush_timeout
#    print("GD-01")
#    if uses_gdrive is True and use_force_flush is True:
#        print("GD-02")
#        if time.time() - flush_timer > flush_timeout:
#            print(f"gd-flush: {epoch} {logs}")
#            flush_timer=time.time()
#            drive.flush_and_unmount()
#            mount_gdrive()


# In[ ]:


def math_train(model, dataset, validation, batch_size=8192, epochs=5000, steps_per_epoch=2000, log_path="./logs"):
    """ Training loop """
    interrupted = 2
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=log_path
        # histogram_freq=1,
        # write_images=1,
        # update_freq='batch'
        )
    # Risky code! Tries to force-flush gdrive via lambda-callback
    lambda_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end = gdrive_flusher
    )
    try:
        if is_tpu:
            model.fit(dataset, validation_data=validation, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[tensorboard_callback, lambda_callback])
            interrupted=0
        else:
            model.fit(dataset, validation_data=validation, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard_callback, lambda_callback])
            interrupted=0
    except KeyboardInterrupt:
        print("")
        print("")
        print("---------INTERRUPT----------")
        print("")
        print("Training interrupted")
        interrupted = 1 # user stopped runtime
    except Exception as e:
        interruped = 2  # Bad: something crashed.
        print(f"INTERNAL ERROR")
        print(f"Exception {e}")
    finally:
        return interrupted


# In[ ]:


def instantiate_models(model_file, model_type, import_weights=True):
    if is_tpu:
        # Generate a second CPU model for testing:
        test_model = get_model(save_path=None, on_tpu=False, model_type=model_type)
        math_model = get_model(save_path=model_file, on_tpu=True, model_type=model_type)
    else:
        test_model = None
        math_model = get_model(save_path=model_file, on_tpu=False, model_type=model_type, import_weights=import_weights)
    return math_model, test_model


# In[ ]:


def do_training(math_model, training_dataset, validation_dataset, math_data, epochs_per_cycle, model_file=None, 
                weights_file=None, test_model=None, cycles=100, steps_per_epoch=1000, reweight_size=1000, valid_ops=None, regenerate_data_after_cycles=0, data_func=None,
                log_path='./logs'):
    # Training
    for mep in range(0, cycles):
        print()
        print()
        print(f"------ Meta-Epoch {mep+1}/{cycles} ------")
        print()
        if regenerate_data_after_cycles!=0 and data_func is not None:
            if mep>0 and (mep+1)%regenerate_data_after_cycles==0:
                math_data, training_dataset, validation_dataset = data_func()
        if mep==0 and is_tpu is True:
            print("There will be some warnings by Tensorflow, documenting some state of internal decoherence, currently they can be ignored.")
        interrupted = math_train(math_model, training_dataset, validation=validation_dataset, epochs=epochs_per_cycle, steps_per_epoch=steps_per_epoch, log_path=log_path)
        if interrupted <2:
            if is_tpu:
                if test_model is None:
                    print("Fatal: tpu-mode needs test_model on CPU")
                    return False
                print("Injecting weights into test_model:")
                test_model.set_weights(math_model.get_weights())
                if weights_file is not None:
                    print(f"Saving test-model weights to {weights_file}")
                    test_model.save_weights(weights_file)
                    print("Done")
                print(f"Checking {reweight_size} datapoints for accuracy...")
                math_data.check_results(test_model, samples=reweight_size, short_math=False, valid_ops=valid_ops, verbose=False)
            else:
                if model_file is not None:
                    print("Saving math-model")
                    math_model.save(model_file)
                    print("Done")
                print(f"Checking {reweight_size} datapoints for accuracy...")
                math_data.check_results(math_model, samples=reweight_size, short_math=False, valid_ops=valid_ops, verbose=False)
        if interrupted>0:
            break


# ## The actual training

# In[ ]:


model_type='conv_recurrent'  # see: model_dict definition.
epochs_per_cycle=250
cycles = 100  # perform 100 cycles, each cycle trains with epochs_per_cycle epochs.
regenerate_data_after_cycles=2  # if !=0, the training data will be created anew after each number of 
                                # regenerace_data_after_cycles cycles. Disadvantage: when training TPU, 
                                # Google might use the time it takes to regenerate to training data to 
                                # terminate your session :-/
samples=2000000  # Number training data examples
batch_size=20000 
valid_ops=None  # Default: None (all ops), or list of ops, e.g. ['*', '/'] trains only multiplication and division.
steps_per_epoch=samples//batch_size  # TPU stuff
model_file, weights_file, cache_stub, log_path = init_paths(model_type=model_type, log_to_gdrive=True)

def train_data(regen=True):
    return get_datasets(pre_weight=True, samples=samples, validation_samples=50000, batch_size=batch_size, short_math=False, 
                                     valid_ops=valid_ops, cache_file_stub=cache_stub, use_cache=True, regenerate_cached_data=regen)
math_data, train, val = train_data(regen=False)
math_model, test_model = instantiate_models(model_file, model_type, import_weights=True)


# In[ ]:


# use the python variable log_path:
get_ipython().run_line_magic('tensorboard', '--logdir "{log_path}"')


# In[ ]:


do_training(math_model, train, val, math_data, epochs_per_cycle, model_file=model_file, 
            weights_file=weights_file, test_model=test_model, cycles=cycles, steps_per_epoch=steps_per_epoch, valid_ops=valid_ops, 
            regenerate_data_after_cycles=regenerate_data_after_cycles, data_func=train_data, log_path=log_path)


# # Testing and applying the trained model

# In[ ]:


if is_tpu is False:
    test_model = math_model
math_data.check_results(test_model, samples=1000, short_math=False, verbose=True)


# In[ ]:


dx,dy,_,_,_=math_data.create_data_point(22,33,'+')


# In[ ]:


math_data.decode_results(test_model.predict(np.array([dx])))


# In[ ]:


def calc(inp):
    args=inp.split(' ')
    if len(args)!=3:
        print("need three space separated tokens: <int> <operator> <int>, e.g. '3 + 4' or '4 XOR 5'")
        return False
    if args[1] not in math_data.model_ops:
        print(f"{args[1]} is not a known operator.")
        return False
    op1=int(args[0])
    op2=int(args[2])
    dx,dy,_,_,_=math_data.create_data_point(op1, op2, args[1])
    ans=math_data.decode_results(test_model.predict(np.array([dx])))
    print(f"{op1} {args[1]} {op2} = {ans[0]}")
    op=f"{op1} {args[1]} {op2}"
    op=op.replace('AND', '&').replace('XOR','^').replace('=','==').replace('OR','|')
    an2=eval(op)
    if ans[0]!=an2:
        print("Error")
        print(bin(ans[0]))
        print(bin(an2))
    return ans[0],an2


# In[ ]:


calc("222 = 223")


# In[ ]:


eval("2333+1120")


# In[ ]:


calc("8812 = 8812")


# In[ ]:


999/27


# In[ ]:


calc("3 * 4")


# In[ ]:


calc ("1 AND 3")


# In[ ]:




