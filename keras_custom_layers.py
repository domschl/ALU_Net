import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import math

class ResidualBlock(layers.Layer):
    def __init__(self, units, highway=False, **kwargs):
        self.units=units
        self.highway=highway
        super(ResidualBlock, self).__init__(**kwargs)
        self.dense1 = layers.Dense(self.units)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(self.units)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'highway': self.highway
        })
        return config

    def call(self, inputs):
        x=self.dense1(inputs)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.dense2(x)
        x=self.bn2(x)
        if self.highway:
            x=self.relu2(x)
            x=x+inputs
        else:
            x=x+inputs
            x=self.relu2(x)
        return x

class ResidualDense(layers.Layer):
    def __init__(self, units, regularizer=0, **kwargs):
        self.units=units
        self.regularizer=regularizer
        super(ResidualDense, self).__init__(**kwargs)
        if self.regularizer != 0:
            self.dense1 = layers.Dense(self.units, kernel_regularizer=keras.regularizers.l2(self.regularizer))
        else:
            self.dense1 = layers.Dense(self.units)       
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'regularizer': self.regularizer
        })
        return config

    def call(self, inputs):
        x=self.dense1(inputs)
        x=self.relu(x)
        x=self.bn1(x)
        x=x+inputs
        return x

class ResidualDenseStack(layers.Layer):
    def __init__(self, units, layer_count, regularizer=0, **kwargs):
        self.units=units
        self.layer_count=layer_count
        self.regularizer=regularizer

        super(ResidualDenseStack, self).__init__(**kwargs)
        self.rd=[]
        for _ in range(0, self.layer_count):
            self.rd.append(ResidualDense(self.units, regularizer=self.regularizer))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layer_count,
            'regularizer': self.regularizer
        })
        return config

    def call(self, inputs):
        x=self.rd[0](inputs)
        for i in range(1, self.layer_count):
            x=self.rd[i](x)
        return x

class ParallelResidualDenseStacks(layers.Layer):
    def __init__(self, units, layer_count, stacks, dispatch, regularizer=0, **kwargs):
        super(ParallelResidualDenseStacks, self).__init__(**kwargs)
        self.units=units
        self.layer_count=layer_count
        self.stacks=stacks
        self.dispatch=dispatch
        self.regularizer=regularizer

        if self.dispatch is True:
            self.scale = layers.Dense(units*stacks, activation=None)
        else:
            self.scale = layers.Dense(units, activation=None)

        self.rds=[]
        for _ in range(0, self.stacks):
            self.rds.append(ResidualDenseStack(self.units, self.layer_count, regularizer=self.regularizer))
        self.rescale_relu = layers.ReLU()
        self.concat = layers.Concatenate()
        if self.regularizer != 0:
            self.rescale = layers.Dense(self.units, kernel_regularizer=keras.regularizers.l2(self.regularizer))
        else:
            self.rescale = layers.Dense(self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layer_count,
            'stacks': self.stacks,
            'dispatch': self.dispatch,
            'regularizer': self.regularizer
        })
        return config

    def call(self, inputs):
        xa=[]
        # Scale up
        x=self.scale(inputs)
        for i in range(0, self.stacks):
            if self.dispatch:
                xa.append(self.rds[i](x[:,i*self.units:(i+1)*self.units]))
            else:
                xa.append(self.rds[i](x))
        x=self.concat(xa)
        x=self.rescale(x)
        x=self.rescale_relu(x)
        return x

class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units=units
        self.pm = layers.Permute((2,1))
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        self.w_keys = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer="random_normal", trainable=True)
        self.w_queries = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer="random_normal", trainable=True)
        self.w_values = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer="random_normal", trainable=True)
        self.scale = self.add_weight(shape=(self.units, input_shape[-1]))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config 

    def call(self, inputs):
        vk = tf.matmul(inputs, self.w_keys)
        vq = tf.matmul(inputs, self.w_queries)
        vv = tf.matmul(inputs, self.w_values)
        kq = tf.matmul(vk, vq, transpose_b=True)/math.sqrt(self.units)
        sm = self.softmax(kq)
        print(f"sm={sm.shape}, vv={vv.shape}")
        x = tf.matmul(sm, self.pm(vv), transpose_b=True)
        out = tf.matmul(x, self.scale)
        return out

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, units, heads, additive=False, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.units=units
        self.heads=heads
        self.additive=additive
        self.mhsa=[]
        for _ in range(0,self.heads):
            self.mhsa.append(SelfAttention(self.units))
        if self.additive is False:
            self.cc = layers.Concatenate(axis=1)
            self.pm = layers.Permute((2,1))

    def build(self, input_shape):
        if self.additive is False:
            self.w_heads = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                        initializer="random_normal", trainable=True)
            self.w_xheads = self.add_weight(shape=(self.heads*input_shape[1], input_shape[1]),
                                        initializer="random_normal", trainable=True)
        else:
            self.w_heads = []
            for _ in range(0, self.heads):
                self.w_heads.append(self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                        initializer="random_normal", trainable=True))
                                    

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'heads': self.heads,
            'additive': self.additive
        })
        return config

    def call(self, inputs):
        if self.additive is True:
            for i in range(0, self.heads):
                if i==0:
                    x=tf.matmul(self.mhsa[i](inputs), self.w_heads[i])
                else:
                    x=x+tf.matmul(self.mhsa[i](inputs), self.w_heads[i])
        else:
            xa=[]
            for i in range(0, self.heads):
                xa.append(self.mhsa[i](inputs))
            x=self.cc(xa)
            x = tf.matmul(x, self.w_heads)
            x = self.pm(x)
            x = tf.matmul(x, self.w_xheads)
            x = self.pm(x)
        return x
