import numpy
import theano
import theano.tensor as T
import pickle

class Convolution:
	def __init__(self, num, dims, wins, mults, maxs):
		self.num = num
		self.dims = dims
		self.wins = wins
		self.mults = mults
		self.maxs = maxs

	@classmethod
	def from_pickle(self, pickle_file):
		#self.num = pickle_file.load()
		#self.dims = pickle_file.load()
		#self.wins = pickle_file.load()
		#self.mults = pickle_file.load()
		#self.maxs = pickle_file.load()
		self = self(pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load() 
			)
		
		self.filters = []		
		self.biases = []		

		for i in range(self.num):
			self.filters.append(theano.shared(pickle_file.load()))
			self.biases.append(theano.shared(pickle_file.load()))
		return self

	def init_random(self):
		self.filters = [theano.shared((
			numpy.random.rand(self.wins[i], self.dims[i], 
			self.maxs[i], self.dims[i+1], self.mults[i])-0.5) * 2.0/numpy.sqrt(self.dims[i] * self.wins[i] * self.mults[i]))
			for i in range(self.num)]
		
		self.biases = [theano.shared((
			numpy.random.rand(self.maxs[i], self.dims[i+1])-0.5))
			for i in range(self.num)]
		
	def dump(self, pickle_file):
		pickle_file.dump(self.num)
		pickle_file.dump(self.dims)
		pickle_file.dump(self.wins)
		pickle_file.dump(self.mults)
		pickle_file.dump(self.maxs)
		
		for i in range(self.num):
			pickle_file.dump(self.filters[i].get_value())
			pickle_file.dump(self.biases[i].get_value())
			

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
			y = T.reshape(y[:, 0:k*wins[i], :], [batch_size, k, wins[i], dims[i]])
			
			y = T.tensordot(y, self.filters[i], [[2, 3], [0, 1]])
			y = T.reshape(y, [batch_size, k, maxs[i], dims[i+1]])

			y = y + T.reshape(self.biases[i], [1, 1, maxs[i], dims[i+1]])

			#y = T.switch(mask_rng.binomial(size=y.shape, p=0.7), y, 0)

			y = y.max(2)
			
		
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

	@classmethod
	def from_pickle(self, pickle_file):
		#self.num = pickle_file.load()
		#self.dims = pickle_file.load()
		#self.wins = pickle_file.load()
		#self.mults = pickle_file.load()
		#self.maxs = pickle_file.load()
		self = self(pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load(), 
			pickle_file.load() 
			)

		self.filters = []		
		self.biases = []		

		for i in range(self.num):
			self.filters.append(theano.shared(pickle_file.load()))
			self.biases.append(theano.shared(pickle_file.load()))
		return self

	def init_random(self):
		self.filters = [theano.shared((
			numpy.random.rand(self.dims[i], 
			self.wins[i], self.maxs[i], self.dims[i+1], self.mults[i])-0.5)*2.0/numpy.sqrt(self.dims[i] * self.wins[i] * self.mults[i]))
			for i in range(self.num)]
		
		self.biases = [theano.shared((
			numpy.random.rand(self.maxs[i], self.dims[i+1])-0.5))
			for i in range(self.num)]
	
	def dump(self, pickle_file):
		pickle_file.dump(self.num)
		pickle_file.dump(self.dims)
		pickle_file.dump(self.wins)
		pickle_file.dump(self.mults)
		pickle_file.dump(self.maxs)
		
		for i in range(self.num):
			pickle_file.dump(self.filters[i].get_value())
			pickle_file.dump(self.biases[i].get_value())

	def apply(self, x):
		num = self.num
		dims = self.dims
		wins = self.wins
		mults = self.mults
		maxs = self.maxs
		

		self.x = x
		batch_size, k, dim0 = T.shape(x)
		y = T.reshape(x, [batch_size, k, dim0])

		for i in range(num):
			
			y = T.tensordot(y, self.filters[i], [[2], [0]])
			y = T.reshape(y, [batch_size, k*wins[i], maxs[i], dims[i+1]])


			k = k * wins[i]
			
			y = y + T.reshape(self.biases[i], [1, 1, maxs[i], dims[i+1]])

			y = y.max(2)
			
		
		self.y = y
		return y
			
	def getParameters(self):
		return self.filters + self.biases
	
	def getGradients(self, cost, mult=1.0):
		return [T.grad(cost, filt)*mult for filt in self.filters] + [T.grad(cost, bias)*mult for bias in self.biases]
