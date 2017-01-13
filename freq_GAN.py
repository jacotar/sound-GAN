import numpy
import theano
import theano.tensor as T
import pickle

import copy

class Discriminator:
	def __init__(self, last_dim, last_nonlin, dim, win, nonlin):
		self.num = len(win)

		self.last_dim = last_dim
		self.last_nonlin = last_nonlin
		self.last_last_dim = last_nonlin.mod_dim(last_dim)

		self.dim = dim
		self.win = win
		self.next_dim = [nonlin[i].mod_dim(dim[i+1]) for i in range(self.num)]
		self.nonlin = nonlin


	def init_random(self):
		num = self.num
		last_last_dim = self.last_last_dim
		dim = self.dim
		win = self.win
		next_dim = self.next_dim
		nonlin = self.nonlin
		
		# 1D convolution
		self.filters = [theano.shared(
				numpy.random.normal(0.0, 0.5/numpy.sqrt(dim[i]), [win[i], dim[i], next_dim[i]])) 
				for i in range(num)]
		self.biases = [
			theano.shared(numpy.zeros(next_dim[i]))
			for i in range(num)]
		
		self.direct_filters = [theano.shared(
				numpy.random.normal(0.0, numpy.power(0.5, i-num)/numpy.sqrt(dim[i+1]), [dim[i+1], last_last_dim])) 
				for i in range(num)]
		self.direct_biases = theano.shared(numpy.zeros(last_last_dim))

	
	def __call__(self, x):
		y = x
		batch_size, k, dim0 = T.shape(x)


		rs = numpy.random.RandomState(1234)
		mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
		
		discr = 0

		self.ys = []
		for i in range(self.num):
			dim = self.dim[i]
			win = self.win[i]
			next_dim = self.next_dim[i]
			
			k = k / win
			y = T.reshape(y[:, 0:k*win, :], [batch_size, k, win, dim])
			y = T.tensordot(y, self.filters[i], [[2, 3], [0, 1]])
			#if i < self.num - 1:
			#	y = T.switch(mask_rng.binomial(size=y.shape, p=0.5), y, 0)
			y = self.nonlin[i](y + self.biases[i])

			discr += T.tensordot(y, self.direct_filters[i], [[2], [0]]).mean([1], keepdims=True)
			

			y.name = "y"+str(i)
			self.ys.append(y)
		
		return self.last_nonlin(discr + self.direct_biases).sum([1])

	def getParameters(self):
		return  self.filters + self.biases + self.direct_filters + [self.direct_biases] 
	
	def getGradients(self, cost, mult=1.0):	
		return [T.grad(cost, param)*mult for param in self.getParameters()] 
	
	def save(self, path):
		with open(path, "wb") as f:
			pickle_file = pickle.Pickler(f)
			
			pickle_file.dump(self.last_dim)	
			pickle_file.dump(self.last_nonlin)	

			pickle_file.dump(self.dim)
			pickle_file.dump(self.win)
			pickle_file.dump(self.nonlin)
		
			for i in range(self.num):
				pickle_file.dump(self.filters[i].get_value())
				pickle_file.dump(self.biases[i].get_value())
				pickle_file.dump(self.direct_filters[i].get_value())
			
			pickle_file.dump(self.direct_biases.get_value())
			
	@classmethod
	def load(self, path):
		with open(path, "rb") as f:
			pickle_file = pickle.Unpickler(f)
	
			self = self(pickle_file.load(),
				pickle_file.load(),
				pickle_file.load(),
				pickle_file.load(),
				pickle_file.load())
		
			self.filters = []
			self.biases = []
			self.direct_filters = []
			#self.direct_biases = []

			for i in range(self.num):
				self.filters.append(theano.shared(pickle_file.load()))
				self.biases.append(theano.shared(pickle_file.load()))
				self.direct_filters.append(theano.shared(pickle_file.load()))
			self.direct_biases = theano.shared(pickle_file.load())
			return self		

		

class Generator:
	def __init__(self, dim, win, nonlin):
		self.num = len(win)
		self.gen_dim = dim[0]

		self.dim = dim
		self.win = win
		self.next_dim = [nonlin[i].mod_dim(dim[i+1]) for i in range(self.num)]
		self.nonlin = nonlin
	
	def size_from(self, n=1):
		return numpy.product(self.win) * n

	def init_random(self):
		num = self.num
		dim = self.dim
		win = self.win
		next_dim = self.next_dim

		# 1D deconvolution
		self.filters = [
			theano.shared(numpy.random.normal(0.0, 1.0/numpy.sqrt(dim[i]), 
				[dim[i], win[i], next_dim[i]]))
			for i in range(num)]
		self.biases = [
			theano.shared(numpy.zeros(next_dim[i]))
			for i in range(num)]

	def __call__(self, z):
		v = z
		k = 1
		self.vs = []
		for i in range(self.num):
			dim = self.dim[i]
			win = self.win[i]
			next_dim = self.next_dim[i]
			
			v = T.tensordot(v, self.filters[i], [[2], [0]])
			v = T.reshape(v, [-1, k*win, next_dim])
			v = self.nonlin[i](v + self.biases[i])
			
			k = k*win
			self.vs.append(v)
		
		return v

	def get_encoder(self, last_nonlin, back_nonlin):
		return Discriminator(self.gen_dim, last_nonlin, 
					list(reversed(self.dim)), 
					list(reversed(self.win)), 
					back_nonlin)

	def ranks(self):
		return [numpy.linalg.matrix_rank(
				numpy.reshape(self.filters[i].get_value(), [self.dim[i], -1])) 
			for i in range(self.num)]

	def getParameters(self):
		return self.filters + self.biases
	
	def getGradients(self, cost, mult=1.0):	
		return [T.grad(cost, param)*mult for param in self.getParameters()] 

	def normL1(self):
		return numpy.mean([numpy.mean(numpy.absolute(param.get_value()), axis=tuple(range(len(param.get_value().shape)))) 
			for param in self.getParameters()])

	def writePickle(self, pickle_file):
		pickle_file.dump(self.dim)
		pickle_file.dump(self.win)
		pickle_file.dump(self.nonlin)
	
		for i in range(self.num):
			pickle_file.dump(self.filters[i].get_value())
			pickle_file.dump(self.biases[i].get_value())
	
	def save(self, path):
		with open(path, "wb") as f:
			self.writePickle(pickle.Pickler(f))	
			
	def fromPickle(self, pickle_file):
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		
		for i in range(self.num):
			self.filters[i].set_value(pickle_file.load())
			self.biases[i].set_value(pickle_file.load())
	
	@classmethod
	def load(self, path):
		with open(path, "rb") as f:
			pickle_file = pickle.Unpickler(f)
	
			self = self(pickle_file.load(),
				pickle_file.load(),
				pickle_file.load())

			self.filters = []
			self.biases = []
	
			for i in range(self.num):
				self.filters.append(theano.shared(pickle_file.load()))
				self.biases.append(theano.shared(pickle_file.load()))
	
			return self		

		
