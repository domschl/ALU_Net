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
    def __init__(self, units, layers, **kwargs):
        self.units=units
        self.layers=layers

        super(ResidualDenseStack, self).__init__(**kwargs)
        self.rd=[]
        for _ in range(0, self.layers):
            self.rd.append(ResidualDense(self.units))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layers
        })
        return config

    def call(self, inputs):
        x=self.rd[0](inputs)
        for i in range(1, self.layers):
            x=self.rd[i](x)
        return x

class ParallelResidualDenseStacks(layers.Layer):
    def __init__(self, units, layers, stacks, dispatch, **kwargs):
        self.units=units
        self.layers=layers
        self.stacks=stacks
        self.dispatch=dispatch

        if self.dispatch is True:
            self.scale = layers.Dense(units*stacks, activation=None)

        super(ParallelResidualDenseStacks, self).__init__(**kwargs)
        self.rds=[]
        for _ in range(0, self.stacks):
            self.rds.append(ResidualDenseStack(self.units, self.layers))

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'layers': self.layers,
            'stacks': self.stacks,
            'dispatch': self.dispatch
        })
        return config

    def call(self, inputs):
        if self.dispatch:
            # Scale up
            x=self.scale(inputs)
        else:
            x=inputs
        for i in range(0, self.stacks):
            if i==0:
                if self.dispatch:
                    x=x[:,i*self.units:(i+1)*self.units]
                else:
                    x=self.rds[0](x)
            else:
                x = x+self.rds[i](inputs)
        x=x+inputs
        return x
