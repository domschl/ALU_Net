import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, units):
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(units)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(units)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, inputs):
        x=self.dense1(inputs)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.dense2(x)
        x=self.bn2(x)
        x=x+inputs
        x=self.relu2(x)
        return x        
