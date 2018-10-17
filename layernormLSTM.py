from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class LayerNormBasicLSTMCell(rnn_cell_impl.RNNCell):
	"""LSTM unit with layer normalization and recurrent dropout.
	This class adds layer normalization and recurrent dropout to a
	basic LSTM unit. Layer normalization implementation is based on:
		https://arxiv.org/abs/1607.06450.
	"Layer Normalization"
	Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
	and is applied before the internal nonlinearities.
	Recurrent dropout is base on:
		https://arxiv.org/abs/1603.05118
	"Recurrent Dropout without Memory Loss"
	Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
	"""

	def __init__(self,
				num_units,
				forget_bias=1.0,
				input_size=None,
				activation=math_ops.tanh,
				layer_norm=True,
				norm_gain=1.0,
				norm_shift=0.0,
				dropout_keep_prob=1.0,
				dropout_prob_seed=None,
				reuse=None):
		"""Initializes the basic LSTM cell.
		Args:
			num_units: int, The number of units in the LSTM cell.
			forget_bias: float, The bias added to forget gates (see above).
			input_size: Deprecated and unused.
			activation: Activation function of the inner states.
			layer_norm: If `True`, layer normalization will be applied.
			norm_gain: float, The layer normalization gain initial value. If
				`layer_norm` has been set to `False`, this argument will be ignored.
			norm_shift: float, The layer normalization shift initial value. If
				`layer_norm` has been set to `False`, this argument will be ignored.
			dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
				recurrent dropout probability value. If float and 1.0, no dropout will
				be applied.
			dropout_prob_seed: (optional) integer, the randomness seed.
			reuse: (optional) Python boolean describing whether to reuse variables
				in an existing scope.  If not `True`, and the existing scope already has
				the given variables, an error is raised.
		"""
		super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse)

		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)

		self._num_units = num_units
		self._activation = activation
		self._forget_bias = forget_bias
		self._keep_prob = dropout_keep_prob
		self._seed = dropout_prob_seed
		self._layer_norm = layer_norm
		self._norm_gain = norm_gain
		self._norm_shift = norm_shift
		self._reuse = reuse

	@property
	def state_size(self):
		return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def _norm(self, inp, scope, dtype=dtypes.float32):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(self._norm_gain)
		beta_init = init_ops.constant_initializer(self._norm_shift)
		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
			vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def _linear(self, args):
		out_size = 4 * self._num_units
		proj_size = args.get_shape()[-1]
		dtype = args.dtype
		weights = vs.get_variable("kernel", [proj_size, out_size], dtype=dtype)
		out = math_ops.matmul(args, weights)
		if not self._layer_norm:
			bias = vs.get_variable("bias", [out_size], dtype=dtype)
			out = nn_ops.bias_add(out, bias)
		return out

	def call(self, inputs, state):
		"""LSTM cell with layer normalization and recurrent dropout."""
		c, h = state
		args = array_ops.concat([inputs, h], 1)
		concat = self._linear(args)
		dtype = args.dtype

		i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
		if self._layer_norm:
			i = self._norm(i, "input", dtype=dtype)
			j = self._norm(j, "transform", dtype=dtype)
			f = self._norm(f, "forget", dtype=dtype)
			o = self._norm(o, "output", dtype=dtype)

		g = self._activation(j)
		if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
			g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

		new_c = (
				c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
		# if self._layer_norm:
		# 	new_c = self._norm(new_c, "state", dtype=dtype)
		new_h = self._activation(new_c) * math_ops.sigmoid(o)

		new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
		return new_h, new_state
