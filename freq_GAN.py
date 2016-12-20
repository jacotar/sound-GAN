import numpy
import theano
import theano.tensor as T
import pickle

import copy

class Discriminator:
	def __init__(self, dim, win, nonlin):
		self.num = len(win)
		self.dim = dim
		self.win = win
		self.next_dim = [nonlin[i].mod_dim(dim[i+1]) for i in range(self.num)]
		self.nonlin = nonlin


	def init_random(self):
		num = self.num
		dim = self.dim
		win = self.win
		next_dim = self.next_dim
		nonlin = self.nonlin
		
		# 1D convolution
		self.filters = [theano.shared(
				numpy.random.normal(0.0, 0.5/numpy.sqrt(dim[i]), [next_dim[i], win[i], dim[i]])) 
				for i in range(num)]
		self.biases = [
			theano.shared(numpy.zeros([next_dim[i], 1, 1]))
			for i in range(num)]
		
		self.direct_filters = [theano.shared(
				numpy.random.normal(0.0, 0.5/numpy.sqrt(dim[i+1]), [dim[i+1]])) 
				for i in range(num)]
		self.direct_biases = [theano.shared(0.0) for i in range(num)]

	
	def __call__(self, x):
		y = x
		dim0, k, batch_size = T.shape(x)


		rs = numpy.random.RandomState(1234)
		mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
		
		discr = 0

		self.ys = []
		for i in range(self.num):
			dim = self.dim[i]
			win = self.win[i]
			next_dim = self.next_dim[i]
			
			k = k / win
			y = T.reshape(y[:, 0:k*win, :], [dim, k, win, batch_size])
			y = T.tensordot(self.filters[i], y, [[2, 1], [0, 2]])
			if i < self.num - 1:
				y = T.switch(mask_rng.binomial(size=y.shape, p=0.5), y, 0)
			y = self.nonlin[i](y + T.reshape(self.biases[i], [-1, 1, 1]))
			
			discr += T.tensordot(self.direct_filters[i], y, [[0], [0]]).mean([0]) + self.direct_biases[i]

			self.ys.append(y)
		
		return T.nnet.nnet.sigmoid(discr)

	def getParameters(self):
		return  self.filters + self.biases + self.direct_filters + self.direct_biases 
	
	def getGradients(self, cost, mult=1.0):	
		return [T.grad(cost, param)*mult for param in self.getParameters()] 
	
	def save(self, path):
		with open(path, "wb") as f:
			pickle_file = pickle.Pickler(f)
	
			pickle_file.dump(self.dim)
			pickle_file.dump(self.win)
			pickle_file.dump(self.nonlin)
		
			for i in range(self.num):
				pickle_file.dump(self.filters[i].get_value())
				pickle_file.dump(self.biases[i].get_value())
				pickle_file.dump(self.direct_filters[i].get_value())
				pickle_file.dump(self.direct_biases[i].get_value())
			
	@classmethod
	def load(self, path):
		with open(path, "rb") as f:
			pickle_file = pickle.Unpickler(f)
	
			self = self(pickle_file.load(),
				pickle_file.load(),
				pickle_file.load())
		
			self.filters = []
			self.biases = []
			self.direct_filters = []
			self.direct_biases = []

			for i in range(self.num):
				self.filters.append(theano.shared(pickle_file.load()))
				self.biases.append(theano.shared(pickle_file.load()))
				self.direct_filters.append(theano.shared(pickle_file.load()))
				self.direct_biases.append(theano.shared(pickle_file.load()))
			return self		

		

class Generator:
	def __init__(self, gen_dim, direct_nonlin, dim, win, nonlin):
		self.num = len(win)
		self.gen_dim = gen_dim
		self.direct_nonlin = direct_nonlin
		self.direct_next_dim = [direct_nonlin[i].mod_dim(dim[i]) for i in range(self.num)]

		self.dim = dim
		self.win = win
		self.next_dim = [nonlin[i].mod_dim(dim[i+1]) for i in range(self.num)]
		self.back_nonlin = [copy.copy(nl) for nl in nonlin]
		self.last_dim = [self.back_nonlin[i].mod_dim(dim[i]) for i in range(self.num)]
		self.nonlin = nonlin
	
	def size_from(self, n=1):
		return numpy.product(self.win) * n

	def init_random(self):
		gen_dim = self.gen_dim
		direct_next_dim = self.direct_next_dim
		num = self.num
		dim = self.dim
		win = self.win
		next_dim = self.next_dim
		nonlin = self.nonlin

		# 1D deconvolution
		self.filters = [
			theano.shared(numpy.random.normal(0.0, 1.0/numpy.sqrt(dim[i]), 
				[next_dim[i], win[i], dim[i]]))
			for i in range(num)]
		self.biases = [
			theano.shared(numpy.zeros(next_dim[i]))
			for i in range(num)]

		self.direct_filters = [
			theano.shared(numpy.random.normal(0.0, 1.0/numpy.sqrt(dim[i]), 
				[direct_next_dim[i], gen_dim]))
			for i in range(num)]
		self.direct_biases = [
			theano.shared(numpy.zeros(direct_next_dim[i]))
			for i in range(num)]

	def init_autoencoder(self):
		num = self.num
		dim = self.dim
		win = self.win
		last_dim = self.last_dim
		nonlin = self.nonlin
		
		# 1D deconvolution
		self.dual_filters = [
			theano.shared(numpy.random.normal(0.0, 1.0/numpy.sqrt(dim[i]), 
				[last_dim[i], win[i], dim[i+1]]))
			for i in range(num)]
		self.dual_biases = [
			theano.shared(numpy.zeros([last_dim[i], 1, 1]))
			for i in range(num)]

	def apply_autoencoder(self, x):
		v = x
		dim0, k, batch_size = T.shape(x)
		
		for i in reversed(range(self.num)):
			dim = self.dim[i+1]
			win = self.win[i]
			next_dim = self.last_dim[i]
			
			k = k / win
			v = T.reshape(v[:, 0:k*win, :], [dim, win, k, -1])
			v = T.tensordot(self.dual_filters[i], v, [[2, 1], [0, 1]])
			v = self.back_nonlin[i](v + T.reshape(self.dual_biases[i], [-1, 1, 1]))
		
		self.z_enc = v
		
		for i in range(self.num):
			dim = self.dim[i+1]
			win = self.win[i]
			next_dim = self.next_dim[i]
			
			v = T.tensordot(self.filters[i], v, [[2], [0]])
			v = v.swapaxes(1, 2)
			v = T.reshape(v, [next_dim, k*win, -1])
			v = self.nonlin[i](v + T.reshape(self.biases[i], [-1, 1, 1]))
			
			k = k*win
		
		return v

	
	def __call__(self, z):
		v = 0
		# dim0, batch_size = T.shape(z)
		k = 1
		self.vs = []
		for i in range(self.num):
			dim = self.dim[i]
			win = self.win[i]
			next_dim = self.next_dim[i]
			
			init = T.tensordot(self.direct_filters[i], z, [[1], [0]]) + T.reshape(self.direct_biases[i], [-1, 1])
			init = self.direct_nonlin[i](init)
			v = v + T.reshape(init, [dim, 1, -1])
			
			v = T.tensordot(self.filters[i], v, [[2], [0]])
			v = v.swapaxes(1, 2)
			v = T.reshape(v, [next_dim, k*win, -1])
			v = self.nonlin[i](v + T.reshape(self.biases[i], [-1, 1, 1]))
			
			k = k*win
			self.vs.append(v)
		
		return v
	
	def ranks(self):
		return [numpy.linalg.matrix_rank(
				numpy.reshape(self.filters[i].get_value(), [-1, self.dim[i]])) 
			for i in range(self.num)]

	def getParameters(self):
		return self.filters + self.biases  + self.direct_filters + self.direct_biases
	
	def getAutoencoderParameters(self):
		return self.dual_filters + self.dual_biases + self.filters + self.biases
	
	def getGradients(self, cost, mult=1.0):	
		return [T.grad(cost, param)*mult for param in self.getParameters()] 
	
	def getAutoencoderGradients(self, cost, mult=1.0):	
		return [T.grad(cost, param)*mult for param in self.getAutoencoderParameters()] 

	def writePickle(self, pickle_file):
		pickle_file.dump(self.gen_dim)
		pickle_file.dump(self.direct_nonlin)
		pickle_file.dump(self.dim)
		pickle_file.dump(self.win)
		pickle_file.dump(self.nonlin)
	
		for i in range(self.num):
			pickle_file.dump(self.filters[i].get_value())
			pickle_file.dump(self.biases[i].get_value())
			pickle_file.dump(self.direct_filters[i].get_value())
			pickle_file.dump(self.direct_biases[i].get_value())
	
	def save(self, path):
		with open(path, "wb") as f:
			self.writePickle(pickle.Pickler(f))	
			
	def fromPickle(self, pickle_file):
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		
		for i in range(self.num):
			self.filters[i].set_value(pickle_file.load())
			self.biases[i].set_value(pickle_file.load())
			self.direct_filters[i].set_value(pickle_file.load())
			self.direct_biases[i].set_value(pickle_file.load())
	
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
			self.direct_biases = []
	
			for i in range(self.num):
				self.filters.append(theano.shared(pickle_file.load()))
				self.biases.append(theano.shared(pickle_file.load()))
				self.direct_filters.append(theano.shared(pickle_file.load()))
				self.direct_biases.append(theano.shared(pickle_file.load()))
	
			return self		

		
