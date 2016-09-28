class Convolution:
	def __init__(num, dims, wins, mults, maxs):
		self.num = num
		self.dims = dims
		self.wins = wins
		self.mults = mults
		self.maxs = maxs

	def init_random():
		self.filters = [theano.shared((
				rng.rand(dims[i] * wins[i], 
						maxs[i] * dims[i+1] * mults[i])-0.5)/(dims[i] * wins[i] * 0.1))
			for i in range(num)]
		
		self.step_filters = [theano.shared(
				numpy.zeros([dims[i] * wins[i], 
						maxs[i] * dims[i+1] * mults[i]], numpy.float64))
			for i in range(num)]
		
		self.biases = [theano.shared((
			rng.rand(maxs[i] * dims[i+1])-0.5))
			for i in range(num)]
		
		self.step_biases = [theano.shared(
			numpy.zeros(maxs[i] * dims[i+1], numpy.float64))
			for i in range(num)]

	def apply(x):
		self.x = x
		batch_size, k = T.shape(x)
		y = x

		for i in range(num):
			k = k / wins[i]
			y = T.reshape(y[:, 0:k*wins[i], :], 
					[batch_size * k, wins[i]*dim[i]])
			
			old_y = T.reshape(T.dot(y, filters[i]), [batch_size, k, maxs[i] * dims[i+1] * mults[i]])

			
			y = numpy.zeros(batch_size, k+mults[i], wins[i]*dim[i])
			for j in range(mults[i]):
				y[:, j:k+j, :] = y[:, j:k+j, :] + old_y[:, :, maxs[i] * dims[i+1] * j : maxs[i] * dims[i+1] * (j+1)]

			k = k + mults[i]

			y = y + biases[i]

			y = T.switch(mask_rng.binomial(size=y.shape, p=0.8), y, 0)

			y = T.reshape(y, [batch_size, k, dim[i+1], maxs[i]])
			
			y = y.max(3)
		
		self.y = y
		return y
			

	def set_grad(cost):
		
		self.grad_filters = T.grad(cost, filters)
		self.grad_biases = T.grad(cost, biases)

		return 
			[(step_filters[i], step_filters[i]*0.7 + grad_filters[i]) for i in range(num)] + 
			[(step_biases[i], step_biases[i]*0.7 + grad_biases[i]) for i in range(num)] + 
			[(filters[i], filters[i] + 0.01 * step_filters[i]) for i in range(num)] + 
			[(biases[i], biases[i] + 0.01 * step_biases[i]) for i in range(num)]
