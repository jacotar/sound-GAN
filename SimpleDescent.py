import numpy
import theano
import theano.tensor as T

class SimpleDescent:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients

		self.gradients_mag = [theano.shared(numpy.ones(self.parameters[i].get_value().shape)) for i in range(self.num)]
		# self.gradients = [T.grad(costs[i], parameters[i]) for i in range(num)]

	def step(self, inputs, outputs, learning_rate=0.1, rec=1.0):
		ups = [(self.gradients_mag[i], 
			self.gradients_mag[i]*rec + (1-rec) * abs(self.gradients_mag[i]))
			for i in range(self.num)]
		ups += [(self.parameters[i], 
			rec*self.parameters[i]+learning_rate * self.gradients[i] / (self.gradients_mag[i] + 0.01)) 
			for i in range(self.num)]
		return theano.function(inputs, outputs, updates=ups)


		

