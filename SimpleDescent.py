import numpy
import theano
import theano.tensor as T

class Grad:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients
	
	def step(self, inputs, outputs, learning_rate=0.1):
		ups = [(self.parameters[i], 
			self.parameters[i] + learning_rate * self.gradients[i]) 
			for i in range(self.num)]
		return theano.function(inputs, outputs, updates=ups)

class CenterReg:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients

		self.avg_param = [theano.shared(numpy.zeros(self.parameters[i].get_value().shape)) for i in range(self.num)]
		# self.gradients = [T.grad(costs[i], parameters[i]) for i in range(num)]

	def step(self, inputs, outputs, learning_rate=0.1, rec=0.9, intensity=0.01):
		ups = [(self.avg_param[i],
			self.avg_param[i]*rec + (1-rec) * self.parameters[i])
			for i in range(self.num)]
		ups += [(self.parameters[i], 
			self.parameters[i]*(1-intensity*learning_rate)
				+ intensity*learning_rate * self.avg_param[i] 
				+ learning_rate * self.gradients[i]) 
			for i in range(self.num)]
		return theano.function(inputs, outputs, updates=ups)


class AdaGrad:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients

		self.variances = [theano.shared(numpy.ones(self.parameters[i].get_value().shape)) for i in range(self.num)]
		# self.gradients = [T.grad(costs[i], parameters[i]) for i in range(num)]

	def step(self, inputs, outputs, learning_rate=0.1, rec=0.9):
		ups = [(self.variances[i],
			self.variances[i]*rec + (1-rec) * abs(self.gradients[i]))
			for i in range(self.num)]
		ups += [(self.parameters[i], 
			self.parameters[i]
				+ learning_rate * self.gradients[i] / (5.0 + self.variances[i])) 
			for i in range(self.num)]
		return theano.function(inputs, outputs, updates=ups)


		
		
class Momentum:
	def __init__(self, parameters, gradients):
		self.num = len(parameters)
		self.parameters = parameters
		self.gradients = gradients

		self.momentum = [theano.shared(numpy.zeros(self.parameters[i].get_value().shape)) for i in range(self.num)]
		# self.gradients = [T.grad(costs[i], parameters[i]) for i in range(num)]

	def step(self, inputs, outputs, learning_rate=0.1, rec=0.8):
		ups = [(self.momentum[i],
			self.momentum[i]*rec + (1-rec) * self.gradients[i])
			for i in range(self.num)]
		ups += [(self.parameters[i], 
			self.parameters[i]
				+ learning_rate * self.momentum[i]) 
			for i in range(self.num)]
		return theano.function(inputs, outputs, updates=ups)


		
