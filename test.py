from keras_custom_layers import *
import numpy as np

sa=MultiHeadSelfAttention(4)
d=np.random.normal(size=(10,3,4))
o=sa(d)
print(o.shape)

