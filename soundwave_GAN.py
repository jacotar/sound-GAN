import numpy
import theano
import theano.tensor as T

from convolution import Convolution
from convolution import Deconvolution

from SimpleDescent import SimpleDescent

import wave

################## initialization ##############
'''
generator = Deconvolution(
		num = 5,
		dims = [150, 100, 50, 25, 7, 2],
		wins = [3, 3, 3, 4, 4],
		mults = [1, 1, 1, 1, 1],
		maxs = [1, 1, 1, 1, 1])
generator.init_random()

discriminator = Convolution(
		num = 5,
		dims = [2, 7, 25, 50, 100, 150], 
		wins = [4, 4, 3, 3, 3],
		mults = [1, 1, 1, 1, 1],
		maxs = [3, 3, 3, 3, 3])
discriminator.init_random()
'''
generator = Deconvolution(
		num = 2,
		dims = [200, 20, 2],
		wins = [4, 4],
		mults = [1, 1],
		maxs = [1, 1])

discriminator = Convolution(
		num = 2,
		dims = [2, 80, 150], 
		wins = [4, 4],
		mults = [1, 1],
		maxs = [4, 4])
'''

generator = Deconvolution(
		num = 1,
		dims = [5, 2],
		wins = [16*8],
		mults = [1],
		maxs = [1])

discriminator = Convolution(
		num = 1,
		dims = [2, 100], 
		wins = [128],
		mults = [1],
		maxs = [4])
'''



discrimins = theano.shared((numpy.random.rand(discriminator.dims[discriminator.num], 1)-0.5)*2.0)
bias_discrimins = theano.shared(0.0)



discrimins = theano.shared((numpy.random.rand(discriminator.dims[discriminator.num], 1)-0.5)*2.0)
bias_discrimins = theano.shared(0.0)


batch_size = 200
sample_size = numpy.product(discriminator.wins[0:discriminator.num])

audio_file = wave.open('but_one_day.wav')

max_sample = audio_file.getnframes() - sample_size


def makeSample(bs):
	xx = numpy.zeros([bs, sample_size, 2])
	for k in range(bs):
		phase = numpy.random.rand(1) * 2 * numpy.pi
		magnitude = numpy.random.normal(0, 1)
		xx[k, :, 1] = [magnitude * numpy.sin(8.0 * i / sample_size + phase) for i in range(sample_size)]
		
		#audio_file.setpos(numpy.random.randint(max_sample))
		#xx[k, :, :] = numpy.fromstring(audio_file.readframes(sample_size), numpy.int16).reshape([sample_size, 2])/ 65536.0
	zz = numpy.random.normal(0, 1, [bs, 1, generator.dims[0]])
	
	return (xx, zz)	

xx, zz = makeSample(batch_size)


# generator.init_random_with_sample(zz)
generator.init_random()
discriminator.init_random()

#generator.filters[0].set_value(xx[0:generator.dims[0], :, :].reshape([generator.dims[0], generator.wins[0]*generator.dims[1]]))
#generator.biases[0].set_value([0.0, 0.0])

################### graph construction ##############
z = T.dtensor3('z')
x_gen = generator.apply(z)

x_in = T.dtensor3('x_in')

x = T.concatenate([x_gen, x_in])
'''
y_gen = discriminator.apply(x_gen)
batch_size, k, dim = T.shape(y_gen)
discr_gen = T.dot(T.reshape(y_gen, [batch_size*k, dim]), discrimins)
discr_gen = T.nnet.sigmoid(T.reshape(discr_gen, [batch_size, k]) + bias_discrimins)

y_in = discriminator.apply(x_in)
batch_size, k, dim = T.shape(y_in)
discr_in = T.dot(T.reshape(y_in, [batch_size*k, dim]), discrimins)
discr_in = T.nnet.sigmoid(T.reshape(discr_in, [batch_size, k]) + bias_discrimins)
'''
y = discriminator.apply(x)
two_batch_size, k, dim = T.shape(y)
discr = T.dot(T.reshape(y, [two_batch_size*k, dim]), discrimins)
discr_gen0 = discr[0:batch_size, :].mean([0, 1])
discr = T.nnet.sigmoid(T.reshape(discr, [two_batch_size, k]) + bias_discrimins)

discr_gen = discr[0:batch_size, :]
discr_in = discr[batch_size:2*batch_size, :]

cost_discr = numpy.log(discr_in+0.001).mean([0, 1]) + numpy.log(1.001 - discr_gen).mean([0, 1])
cost_gen = numpy.log(discr_gen+0.001).mean([0, 1])

discr_in = discr_in.mean(1)
discr_gen = discr_gen.mean(1)

################# set functions ##################
switch = 0.5 #T.exp(3*cost_discr)

param_gen = generator.getParameters()
#grad_gen = generator.getGradients(cost_gen, switch * 3.0)
grad_gen = generator.getGradients(discr_gen0, switch * 3.0)

param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr, (1-switch))

grad_discrimins = T.grad(cost_discr, discrimins) * (1-switch)
grad_bias_discrimins = T.grad(cost_discr, bias_discrimins) * (1-switch)

descent = SimpleDescent(param_gen + param_discr + [discrimins, bias_discrimins], 
			grad_gen + grad_discr + [grad_discrimins, grad_bias_discrimins])

train = descent.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.05, 0.99)

descent_discr = SimpleDescent(param_discr + [discrimins, bias_discrimins], 
			grad_discr + [grad_discrimins, grad_bias_discrimins])
train_discr = descent_discr.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.05)

generate = theano.function([x_in, z], [x_gen, discr_gen])

################################# training ##########################
#batch_size = 150
for i in range(15000):
	
	xx, zz = makeSample(batch_size)
	train_discr(xx, zz)
	
	xx, zz = makeSample(batch_size)
	
	discr_in, discr_gen, cost = train(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost

print "DISCRIMINATOR"

for i in range(150):
	xx, zz = makeSample(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost


import matplotlib.pyplot as P

x_gen, discr_gen = generate(xx, zz)

P.plot(x_gen[0, :, 0])
P.plot(x_gen[0, :, 1])

P.show()

