import numpy
import theano
import theano.tensor as T

from convolution import Convolution
from convolution import Deconvolution

from SimpleDescent import SimpleDescent

################## initialization ##############
'''
generator = Deconvolution(
		num = 5,
		dims = [150, 100, 50, 25, 7, 2],
		wins = [4, 4, 4, 8, 8],
		mults = [1, 1, 1, 1, 1],
		maxs = [3, 3, 3, 3, 3])
generator.init_random()

discriminator = Convolution(
		num = 5,
		dims = [2, 7, 25, 50, 100, 150], 
		wins = [8, 8, 4, 4, 4],
		mults = [1, 1, 1, 1, 1],
		maxs = [3, 3, 3, 3, 3])
discriminator.init_random()
'''
generator = Deconvolution(
		num = 3,
		dims = [100, 50, 10, 2],
		wins = [4, 8, 8],
		mults = [1, 1, 1],
		maxs = [3, 3, 1])
generator.init_random()

discriminator = Convolution(
		num = 3,
		dims = [2, 7, 25, 50], 
		wins = [8, 8, 4],
		mults = [1, 1, 1],
		maxs = [3, 3, 3])
discriminator.init_random()


discrimins = theano.shared((numpy.random.rand(discriminator.dims[discriminator.num], 1)-0.5)*2.0)
bias_discrimins = theano.shared(0.0)


################### graph construction ##############
z = T.dtensor3('z')
x_gen = generator.apply(z)

y_gen = discriminator.apply(x_gen)
batch_size, k, dim = T.shape(y_gen)
discr_gen = T.dot(T.reshape(y_gen, [batch_size*k, dim]), discrimins)
discr_gen = T.nnet.sigmoid(T.reshape(discr_gen, [batch_size, k]) + bias_discrimins)


x_in = T.dtensor3('x_in')

y_in = discriminator.apply(x_in)
batch_size, k, dim = T.shape(y_in)
discr_in = T.dot(T.reshape(y_in, [batch_size*k, dim]), discrimins)
discr_in = T.nnet.sigmoid(T.reshape(discr_in, [batch_size, k]) + bias_discrimins)

cost_discr = numpy.log(discr_in).mean().mean() + numpy.log(1 - discr_gen).mean().mean()
cost_gen = numpy.log(discr_gen).mean().mean()

discr_in = discr_in.mean(1)
discr_gen = discr_gen.mean(1)

################# set functions ##################
switch = T.exp(3*cost_discr)

param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen, switch)

param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr, (1-switch))

grad_discrimins = T.grad(cost_discr, discrimins) * (1-switch)
grad_bias_discrimins = T.grad(cost_discr, bias_discrimins) * (1-switch)

descent = SimpleDescent(param_gen + param_discr + [discrimins, bias_discrimins], 
			grad_gen + grad_discr + [grad_discrimins, grad_bias_discrimins])

train = descent.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.01, 0.9999)

descent_discr = SimpleDescent(param_discr + [discrimins, bias_discrimins], 
			grad_discr + [grad_discrimins, grad_bias_discrimins])
train_discr = descent_discr.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.01)

generate = theano.function([z], [x_gen, discr_gen])

################################# training ##########################

import wave

batch_size = 50
sample_size = numpy.product(discriminator.wins[0:discriminator.num])

audio_file = wave.open('but_one_day.wav')

max_sample = audio_file.getnframes() - sample_size

xx = numpy.zeros([batch_size, sample_size, 2])

for i in range(5000):

	for k in range(batch_size):
		audio_file.setpos(numpy.random.randint(max_sample))
		xx[k, :, :] = numpy.fromstring(audio_file.readframes(sample_size), numpy.int16).reshape([sample_size, 2])/ 65536.0
	zz = numpy.random.normal(0, 1, [batch_size, 1, generator.dims[0]])
	
	discr_in, discr_gen, cost = train(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost


for i in range(300):

	for k in range(batch_size):
		audio_file.setpos(numpy.random.randint(max_sample))
		xx[k, :, :] = numpy.fromstring(audio_file.readframes(sample_size), numpy.int16).reshape([sample_size, 2])/ 65536.0
	zz = numpy.random.normal(0, 1, [batch_size, 1, generator.dims[0]])
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost


