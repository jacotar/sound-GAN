import numpy
import theano
import theano.tensor as T


class Convolution:
	def __init__(self, num, dims, wins, mults, maxs):
		self.num = num
		self.dims = dims
		self.wins = wins
		self.mults = mults
		self.maxs = maxs

	def init_random(self):
		self.filters = [theano.shared((
			numpy.random.rand(self.dims[i] * self.wins[i], 
			self.maxs[i] * self.dims[i+1] * self.mults[i])-0.5)/(self.dims[i] * self.wins[i] * self.mults[i] * 0.6))
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

		rs = numpy.random.RandomState(1234)
		mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))


		self.x = x
		batch_size, k, dim0 = T.shape(x)
		y = x

		for i in range(num):
			k = k / wins[i]
			y = T.reshape(y[:, 0:k*wins[i], :], 
					[batch_size * k, wins[i]*dims[i]])
			
			y = T.reshape(T.dot(y, self.filters[i]), [batch_size, k, maxs[i] * dims[i+1] * mults[i]])

			'''	
			y = T.zeros([batch_size, k+mults[i]-1, wins[i]*dims[i]])
			for j in range(mults[i]):
				y = T.inc_subtensor(y[:, j:k+j, :], old_y[:, :, maxs[i] * dims[i+1] * j : maxs[i] * dims[i+1] * (j+1)])
			'''
			k = k + mults[i] - 1

			y = y + self.biases[i]

			y = T.switch(mask_rng.binomial(size=y.shape, p=0.4), y, 0)

			y = T.reshape(y, [batch_size, k, dims[i+1], maxs[i]])
			
			y = y.max(3)
			
			y = y - y.mean([0, 1], keepdims=True)
				
			#y = y / (T.sqr(y).mean([0, 1], keepdims=True).sqrt() + 0.1)
			y = y / (abs(y).mean([0, 1], keepdims=True) + 0.1)
		
		self.y = y
		return y
			
	def getParameters(self):
		return self.filters + self.biases
	
	def getGradients(self, cost, mult=1.0):
		return [T.grad(cost, filt)*mult for filt in self.filters] + [T.grad(cost, bias)*mult for bias in self.biases]



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
			self.wins[i] * self.maxs[i] * self.dims[i+1] * self.mults[i])-0.5)/(self.dims[i] * self.wins[i] * self.mults[i] * 0.3))
			for i in range(self.num)]
		
		self.biases = [theano.shared((
			numpy.random.rand(self.maxs[i] * self.dims[i+1])-0.5))
			for i in range(self.num)]
	def init_random_with_sample(self, x):
		num = self.num
		dims = self.dims
		wins = self.wins
		mults = self.mults
		maxs = self.maxs
			
		self.filters = []
		self.bias = []
	
		batch_size, k, dim0 = T.shape(x)
		y = T.reshape(x, [batch_size * k, dim0])
		
		for i in range(num):
			filt = numpy.random.rand(dims[i], wins[i] * maxs[i] * dims[i+1] * mults[i])-0.5
			bias = numpy.random.rand(maxs[i] * dims[i+1]) - 0.5
			x = numpy.reshape(numpy.dot(y, filt), [batch_size, k*wins[i], maxs[i] * dims[i+1] * mults[i]])
			k = (k + mults[i] - 1) * wins[i]
			
			y = y + bias

			#y = T.switch(mask_rng.binomial(size=y.shape, p=0.8), y, 0)

			y = T.reshape(y, [batch_size, k, maxs[i], dims[i+1]])
			
			y = y.max(2)

			exp = numpy.mean(y, [0, 1], keepdims=True);
			var = 0.2 + numpy.square(numpy.mean(numpy.square(y - exp), [0, 1], keepdims=True))

			bias = numpy.reshape(bias, [1, maxs[i], dims[i+1]]) / var - exp
			filt = numpy.reshape(filt, [dims[i]*mults[i]*wins[i], maxs[i], dims[i+1]]) / var
			
			self.filters[i] = theano.shared(numpy.reshape(filt, [dims[i], wins[i]*mults[i]*maxs[i]*wins[i] * dims[i+1]]))
			self.biases[i] = theano.shared(numpy.reshape(bias, [maxs[i]*dims[i+1]]))

		self.y = y
		return y
			
	
	def apply(self, x):
		num = self.num
		dims = self.dims
		wins = self.wins
		mults = self.mults
		maxs = self.maxs
		
		rs = numpy.random.RandomState(1234)
		mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))


		self.x = x
		batch_size, k, dim0 = T.shape(x)
		y = T.reshape(x, [batch_size * k, dim0])

		for i in range(num):
			#k = k / wins[i]
			#y = T.reshape(y[:, 0:k*wins[i], :], 
			#		[batch_size * k, wins[i]*dim[i]])
			
			old_y = T.reshape(T.dot(y, self.filters[i]), 
				[batch_size, k * wins[i], 
				maxs[i] * dims[i+1], mults[i]])

			
			y = T.zeros([batch_size, wins[i] * (k + mults[i] - 1), dims[i+1] * maxs[i]])
			for j in range(mults[i]):
				y = T.inc_subtensor(y[:, j*wins[i]:(k+j)*wins[i], :], 
					old_y[:, :, :, j])
			
			

			k = (k + mults[i] - 1) * wins[i]
			
			y = y + self.biases[i]

			#y = T.switch(mask_rng.binomial(size=y.shape, p=0.8), y, 0)

			y = T.reshape(y, [batch_size, k, maxs[i], dims[i+1]])
			
			y = y.max(2)
			
			if i < num-1:
				y = y - y.mean([0, 1], keepdims=True)
				#y = y / (T.sqr(y).mean([0, 1], keepdims=True).sqrt() + 0.1)
				y = y / (abs(y).mean([0, 1], keepdims=True) + 0.1)
		
		
		self.y = y
		return y
			
	def getParameters(self):
		return self.filters + self.biases
	
	def getGradients(self, cost, mult=1.0):
		return [T.grad(cost, filt)*mult for filt in self.filters] + [T.grad(cost, bias)*mult for bias in self.biases]
