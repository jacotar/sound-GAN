import theano.tensor as T

class Function:
	def __init__(self, f):
		self.f = f
	
	def mod_dim(self, d):
		return d
	
	def __call__(self, x):
		return self.f(x)

class Multiplier:
	def __init__(self, f):
		self.f = f
	
	def mod_dim(self, d):
		return d+1
	
	def __call__(self, x):
		q = self.f(x[0, :])

class Maxs:
	def __init__(self, k):
		self.k = k
	
	def mod_dim(self, d):
		self.dim = d
		return d*self.k

	def __call__(self, x):
		sh = T.shape(x)
		x = T.reshape(x, [self.dim, self.k, sh[1], sh[2]])
		x = x.max(1)
		#x = T.reshape(x, [self.dim, sh[1], sh[2]])
		return x

class MaxsZero:
	def __init__(self, k):
		self.k = k
	
	def mod_dim(self, d):
		self.dim = d
		return d*self.k

	def __call__(self, x):
		sh = T.shape(x)
		x = T.reshape(x, [self.dim, self.k, sh[1], sh[2]])
		x = x.max(1)
		x = T.nnet.nnet.relu(x)
		return x

class Identity:
	def __init__(self):
		pass

	def mod_dim(self, d):
		return d
	
	def __call__(self, x):
		return x
