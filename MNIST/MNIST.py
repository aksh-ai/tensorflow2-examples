import tensorflow as tf
from layers import *

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class CNN(tf.Module):
	def __init__(self):
		self.conv1 = Conv2D(1, 64, (3, 3), (1, 1), activation=tf.nn.selu, name='conv1')
		self.conv2 = Conv2D(64, 128, (3, 3), (1, 1), activation=tf.nn.selu, name='conv2')
		self.conv3 = Conv2D(128, 256, (3, 3), (1, 1), activation=tf.nn.selu, name='conv3')
		self.conv4 = Conv2D(256, 512, (3, 3), (1, 1), activation=tf.nn.selu, name='conv4')

		self.fc1 = Dense(147968, 128, activation=tf.nn.selu, name='fc1')
		self.fc2 = Dense(128, 256, activation=tf.nn.selu, name='fc2')
		self.fc3 = Dense(256, 10, activation=tf.nn.softmax, name='out')

		self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2, self.fc3]

		self.params = []

		for layer in self.layers:
			try:
				self.params.append([layer.W, layer.b])
			except:
				pass

		self.params = [j for i in self.params for j in i]

		self.loss = tf.keras.losses.categorical_crossentropy
		self.opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

	@tf.function
	def __call__(self, X: tf.Tensor) -> tf.Tensor:
		X = self.conv1(X)
		X = tf.nn.max_pool2d(X, 2, 1, 'VALID')
		X = self.conv2(X)
		X = tf.nn.max_pool2d(X, 2, 1, 'VALID')
		X = self.conv3(X)
		X = tf.nn.max_pool2d(X, 2, 1, 'VALID')
		X = self.conv4(X)

		X = tf.reshape(X, [-1, X.shape[1] * X.shape[2] * X.shape[3]])

		X = self.fc1(X)
		X = tf.nn.dropout(X, rate=0.2)
		X = self.fc2(X)
		X = tf.nn.dropout(X, rate=0.2)
		X = self.fc3(X)

		return X

	def grad(self, y_hat, y_true):
		with tf.GradientTape() as g:
			y_hat = self.__call__(y_hat)
			error = self.cost(y_hat, y_true)

		return g.gradient(error, self.params), error, y_hat

	def cost(self, y_hat, y_true):
		return tf.reduce_mean(self.loss(y_true, y_hat))

	def backward(self, inputs, targets):
		grads, loss, y_hat = self.grad(inputs, targets)
		self.optimize(grads)
		return loss, y_hat

	def optimize(self, grads):
		self.opt.apply_gradients(zip(grads, self.params))

	def accuracy(self, y, yhat):
		correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))

		num = tf.reduce_sum(tf.cast(correct, dtype = tf.float32))
		den = tf.cast(y.shape[0], dtype = tf.float32)
		return num / den

	def evaluate(self, X_test, y_test, batch_size):
		num_examples = len(y_test)
		acc, loss = 0, 0

		for offset in range(0, num_examples, batch_size):
			end = offset + batch_size
			batch_x, batch_y = X_test[offset:end], y_test[offset:end]

			y_hat = self.__call__(batch_x)
			loss = self.cost(y_hat, batch_y)

			acc = self.accuracy(batch_y, y_hat)

		print(f"Validation Accuracy: {acc:.4f} | Validation Loss: {loss:.4f}\n")

		return acc.numpy(), loss.numpy()

	def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_data=()):
		num_examples = len(y_train)

		total_accuracy = []
		total_loss = []
		loss, acc = 0, 0
		val_loss, val_acc = 0, 0
		trn_err, trn_acy = [], []
		val_err, val_acy = [], []

		batch_x, batch_y = None, None

		X_test, y_test = validation_data

		for i in range(epochs):
			trn_loss = []
			trn_accy = []

			print("Epoch {}".format(i+1))

			for offset in range(0, num_examples, batch_size):
				end = offset + batch_size
				batch_x, batch_y = X_train[offset:end], y_train[offset:end]

				loss, y_hat = self.backward(batch_x, batch_y)

				acc = self.accuracy(batch_y, y_hat)

				if(end==batch_size or end % 8192==0 or end==num_examples):
					print(f"Batch [{end:5d}/{num_examples}] | Accuracy: {acc:.4f} | Loss: {loss:.4f}")

				trn_loss.append(loss)
				trn_accy.append(acc)

			trn_err = tf.reduce_mean(trn_loss).numpy()
			trn_acy = tf.reduce_mean(trn_accy).numpy()

			total_accuracy.append(trn_acy)
			total_loss.append(trn_err)

			val_acc, val_loss = self.evaluate(X_test, y_test, batch_size)

			val_acy.append(val_acc)
			val_err.append(val_loss)

		return {'accuracy': total_accuracy, 'loss': total_loss, 'val_accuracy': val_acy, 'val_loss': val_err}

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(X_train.shape)
print(y_train.shape)

X_train = tf.expand_dims(tf.cast(tf.convert_to_tensor(X_train/255.0), tf.float32), 3)
y_train = tf.keras.utils.to_categorical(y_train)
y_train = tf.cast(tf.convert_to_tensor(y_train), tf.float32)

X_test = tf.expand_dims(tf.cast(tf.convert_to_tensor(X_test/255.0), tf.float32), 3)
y_test = tf.keras.utils.to_categorical(y_test)
y_test = tf.cast(tf.convert_to_tensor(y_test), tf.float32)

model = CNN()

h = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

tf.saved_model.save(model, 'models')