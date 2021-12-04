import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

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
        self.units=units
        self.layer_count=layer_count
        self.stacks=stacks
        self.dispatch=dispatch

        if self.dispatch is True:
            self.scale = layers.Dense(units*stacks, activation=None)

        self.regularizer=regularizer
        super(ParallelResidualDenseStacks, self).__init__(**kwargs)
        self.rds=[]
        for _ in range(0, self.stacks):
            self.rds.append(ResidualDenseStack(self.units, self.layer_count, regularizer=self.regularizer))
        self.relu = layers.ReLU()
        self.concat = layers.Concatenate()
        if self.regularizer != 0:
            self.dense = layers.Dense(self.units, kernel_regularizer=keras.regularizers.l2(self.regularizer))
        else:
            self.dense = layers.Dense(self.units)

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
        if self.dispatch:
            # Scale up
            x=self.scale(inputs)
        else:
            x=inputs
        for i in range(0, self.stacks):
            if i==0:
                if self.dispatch:
                    xa.append(self.rds[0](x[:,i*self.units:(i+1)*self.units]))
                else:
                    xa.append(self.rds[0](x))
            else:
                xa.append(self.rds[i](inputs))
        x=self.concat(xa)
        x=self.dense(x)
        x=self.relu(x)
        return x
