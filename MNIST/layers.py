import tensorflow as tf
from typing import Callable, Iterable, List, Tuple
from tensorflow.keras.initializers import GlorotUniform

class Dense(object):
	def __init__(self, in_feat: int, out_feat: int, activation: Callable = None, name: str = 'dense', initializer: Callable = GlorotUniform(seed=0))->None:
		# Weight parameters
		self.W = tf.Variable(initializer([in_feat, out_feat]), name=name + '_W')
		self.b = tf.Variable(tf.zeros([out_feat]), name=name + '_b')

		# params for weight updates
		self.params = [self.W, self.b]

		# layer attributes
		self.activation = activation

	def __call__(self, X: tf.Tensor) -> tf.Tensor:
		if callable(self.activation):
			return self.activation(tf.add(tf.matmul(X, self.W), self.b))

		else:
			return tf.add(tf.matmul(X, self.W), self.b)

	def forward(self, X: tf.Tensor) -> tf.Tensor:
		return self.__call__(X)

class Conv2D(object):
	def __init__(self, in_feat: int, out_feat: int, kernel_size: Tuple, strides: Tuple = (1, 1), activation: Callable = None, name: str = 'conv', padding: str = 'VALID', use_batch_norm: bool = False, initializer: Callable = GlorotUniform(seed=0)) -> None:
		# Weight parameters
		self.W = tf.Variable(initializer([kernel_size[0], kernel_size[1], in_feat, out_feat]), name=name + '_W')
		self.b = tf.Variable(tf.zeros([out_feat]), name=name + '_b')

		# params for weight updates
		self.params = [self.W, self.b]

		# layer attributes
		self.name = name
		self.stride = strides
		self.padding = padding
		self.activation = activation
		self.use_batch_norm = use_batch_norm

	def __call__(self, X: tf.Tensor) -> tf.Tensor:
		out = tf.nn.conv2d(X, self.W, strides=[1, self.stride[0], self.stride[1], 1], padding=self.padding)
		out = tf.nn.bias_add(out, self.b)

		if self.use_batch_norm:
			mean, variance = tf.nn.moments(out, axes=[0], keepdims=True)
			out = tf.nn.batch_normalization(out, mean=mean, variance=variance, offset=0.0, scale=1.0, variance_epsilon=1e-5)

		if callable(self.activation):
			out = self.activation(out)

		return out

	def forward(self, X: tf.Tensor) -> tf.Tensor:
		return self.__call__(X)