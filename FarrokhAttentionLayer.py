"""
Attention layer implementation by Farrokh Mehryary <farmeh@utu.fi>

Based on Zhou et al. ( https://www.aclweb.org/anthology/P/P16/P16-2034.pdf )

Extensions by Samuel Ronnqvist (commented)

source: https://github.com/sronnqvist/discourse-ablstm
"""

from keras.engine import Layer
from keras import backend as K
import numpy as np

class FarATTN(Layer):
    def __init__(self, return_sequence=False, return_alpha=False, weights=None, **kwargs):
        self.return_sequences = False
        self.supports_masking = False
        self.return_alpha = return_alpha # Switch to make layer ouput alpha weights instead of weighted vector sum
        super(FarATTN, self).__init__(**kwargs)

    def build(self, input_shape):
        LSTM_output_dim = input_shape[2]
        self.W = K.variable(np.random.random((LSTM_output_dim,))*0.2+0.4, name = "W")
        #self.W = K.variable(np.random.random((LSTM_output_dim,)), name = "W")
        self.trainable_weights = [self.W]

    def get_output_shape_for(self, input_shape):
        return (None,input_shape[2])

    def call(self, x, mask=None):
        M = K.tanh(x)
        alpha = K.softmax (K.dot(M, self.W) )
        # Modify behavior to return weights instead of weighted output
        if self.return_alpha:
            return alpha
        else:
            #return K.tanh(K.dot(alpha,x))[0]
            return K.dot(alpha,x)[0] # Without final tanh, as in Zhou et al.

    def compute_mask(self, inputs, masks=None):
        return None
