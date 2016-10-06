import numpy
import theano
import theano.tensor as T

class SimpleDescent:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients
		# self.gradients = [T.grad(costs[i], parameters[i]) for i in range(num)]

	def step(self, inputs, outputs, learning_rate=0.1, rec=1.0):
		return theano.function(inputs, outputs, 
			updates = [(self.parameters[i], rec*self.parameters[i]+learning_rate * self.gradients[i]) 
			for i in range(self.num)])


		

