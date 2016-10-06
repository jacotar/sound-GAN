import numpy
import theano
import theano.tensor as T

class Deconvolution:
	def __init__(self, num, dims, wins, mults, maxs):
		self.num = num
		self.dims = dims
		self.wins = wins
		self.mults = mults
		self.maxs = maxs

	def init_random(self):
		self.filters = [theano.shared((
			numpy.random.rand(self.dims[i], 
			self.wins[i] * self.maxs[i] * self.dims[i+1] * self.mults[i])-0.5)/(self.dims[i] * self.wins[i] * 0.1))
			for i in range(self.num)]
		
		self.biases = [theano.shared((
			numpy.random.rand(self.maxs[i] * self.dims[i+1])-0.5))
			for i in range(self.num)]
	
	def apply(self, x):
		num = self.num
		dims = self.dims
		wins = self.wins
		mults = self.mults
		maxs = self.maxs

		self.x = x
		batch_size, k, dim0 = T.shape(x)
		y = T.reshape(x, [batch_size * k, dim0])

		for i in range(num):
			#k = k / wins[i]
			#y = T.reshape(y[:, 0:k*wins[i], :], 
			#		[batch_size * k, wins[i]*dim[i]])
			
			old_y = T.reshape(T.dot(y, self.filters[i]), 
				[batch_size, k * wins[i], 
				maxs[i] * dims[i+1] * mults[i]])

			
			y = T.zeros([batch_size, wins[i] * (k + mults[i]), dims[i+1] * maxs[i]])
			for j in range(mults[i]):
				y[:, j*wins[i]:(k+j)*wins[i], :] += old_y[:, :, maxs[i] * dims[i+1] * j : maxs[i] * dims[i+1] * (j+1)]

			k = (k + mults[i]) * wins[i]

			y = y + self.biases[i]

			# y = T.switch(mask_rng.binomial(size=y.shape, p=0.8), y, 0)

			y = T.reshape(y, [batch_size, k, dim[i+1], maxs[i]])
			
			y = y.max(3)
		
		self.y = y
		return y
			
	def getParameters(self):
		return self.filters + self.biases
	
	def getGradients(self, cost):
		return [T.grad(cost, filt) for filt in self.filters] + [T.grad(cost, bias) for bias in self.biases]
