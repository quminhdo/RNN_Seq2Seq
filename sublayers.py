import tensorflow as tf
import utils

class Layer_Norm:
    def __init__(self):
        self.gain = tf.get_variable(name="norm_gain", initializer=1.0)
        self.bias = tf.get_variable(name="norm_bias", initializer=0.0)
        self.epsilon = 1.0e-12

    def __call__(self, inputs):
        mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) * tf.rsqrt(var + self.epsilon)
        output = self.gain * normalized + self.bias
        return output
        
        
