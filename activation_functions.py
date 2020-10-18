from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.regularizers import l2
from keras import optimizers
from keras.models import Model
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class Psqish_whole(Layer):

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Psqish_whole, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Psqish_whole, self).build(input_shape)

    def call(self, inputs, mask=None):
        return sqish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Psqish_whole, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def sqish(x,beta=1.0):
  a=tf.clip_by_value(x,-2,200)
  at = a + (a*a)/32.0
  t = a + (a*a)/2.0
  cond = tf.equal(beta, tf.constant(0.0))
  return tf.where(cond,x,(beta * (tf.maximum(0.0, at) + tf.minimum(0.0,t))
  
  
  
  class Psqish_neg(Layer):

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Psqish_neg, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Psqish_neg, self).build(input_shape)

    def call(self, inputs, mask=None):
        return sqish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Psqish_neg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def sqish(x,beta=1.0):
  a=tf.clip_by_value(x,-2,200)
  at = a + (a*a)/32.0
  t = a + (a*a)/2.0
  return tf.maximum(0.0, at) + beta * (tf.minimum(0.0,t))



class Msu(Layer):

    def __init__(self, beta=1.0, trainable=True, **kwargs):
        super(Msu, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Msu, self).build(input_shape)

    def call(self, inputs, mask=None):
        return msu(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Msu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def msu(x,beta=1.0):
  M = 2.0
  orig = x
  cond1 = (-(M/2.0)) - beta
  cond2 = (M/2.0) - beta
  psq_fo = ((tf.math.square(M/2.0 + orig +beta))/(2.0*M)) - beta
  x = tf.where(orig < cond1, 0.0*x - beta, x)
  x = tf.where(tf.logical_and(cond1 <= orig, orig <=cond2), (psq_fo), x)
  y1 = tf.where(orig > cond2, (orig) , x)
  y2 = (tf.maximum(0.0, orig) + beta *tf.minimum(0.0, orig + (orig * orig)/2.0))
  return tf.where(beta <= 1.0, y1, y2)
