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
    def __init__(self, units, **kwargs):
        self.units=units
        super(ResidualDense, self).__init__(**kwargs)
        self.dense1 = layers.Dense(self.units)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config

    def call(self, inputs):
        x=self.dense1(inputs)
        x=self.relu(x)
        x=self.bn1(x)
        x=x+inputs
        return x

class ResidualDenseStack(layers.Layer):
    def __init__(self, units, layer_count, **kwargs):
        self.units=units
        self.layer_count=layer_count

        super(ResidualDenseStack, self).__init__(**kwargs)
        self.rd=[]
        for _ in range(0, self.layer_count):
            self.rd.append(ResidualDense(self.units))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layer_count
        })
        return config

    def call(self, inputs):
        x=self.rd[0](inputs)
        for i in range(1, self.layer_count):
            x=self.rd[i](x)
        return x

class ParallelResidualDenseStacks(layers.Layer):
    def __init__(self, units, layer_count, stacks, **kwargs):
        self.units=units
        self.layer_count=layer_count
        self.stacks=stacks
        super(ParallelResidualDenseStacks, self).__init__(**kwargs)
        self.rds=[]
        for _ in range(0, self.stacks):
            self.rds.append(ResidualDenseStack(self.units, self.layer_count))
        self.relu = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layer_count,
            'stacks': self.stacks
        })
        return config

    def call(self, inputs):
        x=self.rds[0](inputs)
        for i in range(1, self.stacks):
            x = x+self.rds[i](inputs)
        x=self.relu(x)
        # x=x+inputs
        return x
