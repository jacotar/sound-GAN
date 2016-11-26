import numpy
import scipy.spatial.distance as spatial
import theano
import theano.tensor as T

from convolution import Convolution
from convolution import Deconvolution

import SimpleDescent

import wave

import sys
import pickle

################## initialization ##############
gen_dim = 300

if len(sys.argv) <= 1:
	'''
	generator = Deconvolution(
		num = 5,
		dims = [900, 500, 200, 100, 20, 2],
		wins = [5, 5, 5, 16, 16],
		mults = [1, 1, 1, 1, 1],
		maxs = [4, 4, 4, 2, 1])
	encoder = Convolution(
		num = 5,
		dims = [2, 20, 100, 200, 500, 900], 
		wins = [16, 16, 5, 5, 5],
		mults = [1, 1, 1, 1, 1],
		maxs = [1, 2, 4, 4, 4])

	discriminator = Convolution(
		num = 5,
		dims = [2, 20, 100, 200, 500, 500], 
		wins = [16, 16, 5, 5, 5],
		mults = [1, 1, 1, 1, 1],
		maxs = [3, 3, 3, 3, 3])
	'''
	generator = Deconvolution(
		num = 2,
		dims = [100, 150, 2],
		wins = [5, 5],
		mults = [1, 1],
		maxs = [4, 1])

	encoder = Convolution(
		num = 2,
		dims = [2, 150, 100],
		wins = [5, 5],
		mults = [1, 1],
		maxs = [1, 4])

	discriminator = Convolution(
		num = 2,
		dims = [2, 50, 100], 
		wins = [5, 5],
		mults = [1, 1],
		maxs = [4, 4])

	'''
	generator = Deconvolution(
		num = 1,
		dims = [50, 2],
		wins = [16],
		mults = [1],
		maxs = [1])

	encoder = Convolution(
		num = 1,
		dims = [2, 50],
		wins = [16],
		mults = [1],
		maxs = [1])

	discriminator = Convolution(
		num = 1,
		dims = [2, 20], 
		wins = [16],
		mults = [1],
		maxs = [4])

	'''
	#generator = Deconvolution(num=0, dims = [2], wins = [], mults = [], maxs = [])
	#encoder = Convolution(num=0, dims = [2], wins = [], mults = [], maxs = [])
	#discriminator = Convolution(num=0, dims = [2], wins = [], mults = [], maxs = [])

	# generator.init_random_with_sample(zz)
	generator.init_random()
	encoder.init_random()
	discriminator.init_random()

	initiates = theano.shared((numpy.random.rand(gen_dim, generator.dims[0])-0.5) * 2.0 / numpy.sqrt(gen_dim))
	bias_initiates = theano.shared(numpy.zeros([generator.dims[0]]))

	encodes = theano.shared((numpy.random.rand(generator.dims[0], gen_dim)-0.5) * 2.0 / numpy.sqrt(generator.dims[0]))
	bias_encodes = theano.shared(numpy.zeros([gen_dim]))

	discrimins = theano.shared((numpy.random.rand(discriminator.dims[discriminator.num], 1)-0.5) * 2.0 / numpy.sqrt(discriminator.dims[discriminator.num]))
	bias_discrimins = theano.shared(0.0)
else:
	with open(sys.argv[1], "rb") as f:
		pickle_file = pickle.Unpickler(f)

		initiates = theano.shared(pickle_file.load())
		bias_initiates = theano.shared(pickle_file.load())
		encodes = theano.shared(pickle_file.load())
		bias_encodes = theano.shared(pickle_file.load())
		discrimins = theano.shared(pickle_file.load())
		bias_discrimins = theano.shared(pickle_file.load())
		
		generator = Deconvolution.from_pickle(pickle_file)
		encoder = Convolution.from_pickle(pickle_file)
		discriminator = Convolution.from_pickle(pickle_file)


batch_size = 300
sample_size = numpy.product(discriminator.wins[0:discriminator.num])

audio_file = wave.open('but_one_day.wav')

max_sample = audio_file.getnframes() - sample_size


def makeSample(bs):
	xx = numpy.zeros([bs, sample_size, 2])
	for k in range(bs):
		#phase = numpy.random.rand(1) * 2 * numpy.pi
		#magnitude = numpy.random.normal(0, 1)
		#xx[k, :, 1] = [magnitude * numpy.sin(8.0 * i / sample_size + phase) for i in range(sample_size)]
	
		audio_file.setpos(numpy.random.randint(max_sample))
		xx[k, :, :] = numpy.fromstring(audio_file.readframes(sample_size), numpy.int16).reshape([sample_size, 2]) * 4.0 / 65536.0
	zz = numpy.random.normal(0, 1, [bs, gen_dim])
	
	return (xx, zz)	


def energyDst(xs, ys):
	dst = 2*numpy.mean(spatial.cdist(xs, ys, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(xs, xs, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(ys, ys, 'euclidean'))
	return dst
	
xx, zz = makeSample(batch_size)

################### AUTOENCODER ###################

x_enc = T.dtensor3('x_enc')

z0_enc = encoder.apply(x_enc)
z_enc = T.tensordot(z0_enc, encodes, [[2], [0]]) + T.reshape(bias_encodes, [1, 1, gen_dim])

zz0_enc = T.tensordot(z_enc, initiates, [[2], [0]]) + T.reshape(bias_initiates, [1, 1, generator.dims[0]])
zz0_enc = T.nnet.relu(zz0_enc)
x_gen_enc = generator.apply(zz0_enc)

mean_enc = z_enc.mean([0], keepdims=True)
var_enc = T.sqr(z_enc - mean_enc).mean([0, 1])

cost_enc = -T.sqr(x_gen_enc - x_enc).mean([0, 1, 2])
cost_enc += -T.sqr(mean_enc).mean([0, 1, 2])*0.001 - T.sqr(var_enc - 1.0).mean()*0.001


################### graph construction ##############
z = T.dmatrix('z')
z0 = T.tensordot(z, initiates, [[1], [0]]) + T.reshape(bias_initiates, [1, generator.dims[0]])
z0 = T.nnet.relu(z0)
z0 = T.reshape(z0, [-1, 1, generator.dims[0]])
x_gen = generator.apply(z0)

x_in = T.dtensor3('x_in')

x = T.concatenate([x_gen, x_in])

y = discriminator.apply(x)

two_batch_size, k, dim = T.shape(y)
discr = T.tensordot(y, discrimins, [[2], [0]])
discr = T.nnet.sigmoid(T.reshape(discr, [two_batch_size, k]) + bias_discrimins)

discr_gen = discr[0:batch_size, :]
discr_in = discr[batch_size:2*batch_size, :]

cost_discr = numpy.log(discr_in).mean([0, 1]) + numpy.log(1 - discr_gen).mean([0, 1])
# cost_gen = numpy.log(discr_gen+0.001).mean([0, 1])

discr_in = discr_in.mean(1)
discr_gen = discr_gen.mean(1)

mean_in = y[0:batch_size, :, :].mean([0, 1], keepdims=True) 
mean_gen = y[batch_size:2*batch_size, :, :].mean([0, 1], keepdims=True)
var_in = T.sqr(y[0:batch_size, :, :] - mean_in).mean([0, 1])
var_gen = T.sqr(y[batch_size:2*batch_size, :, :] - mean_gen).mean([0, 1])
cost_gen = -T.sqr(mean_in - mean_gen).sum([0, 1, 2]) - 0.7 * T.sqr(var_in - var_gen).sum()

################# set functions ##################

param_enc = encoder.getParameters()
grad_enc = encoder.getGradients(cost_enc, 1.0)
grad_initiates_enc = T.grad(cost_enc, initiates)
grad_bias_initiates_enc = T.grad(cost_enc, bias_initiates)

grad_gen_enc = generator.getGradients(cost_enc, 1.0)
grad_encodes = T.grad(cost_enc, encodes)
grad_bias_encodes = T.grad(cost_enc, bias_encodes)



param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen, 1.0)

grad_initiates = T.grad(cost_gen, initiates)
grad_bias_initiates = T.grad(cost_gen, bias_initiates)


param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr, 1)

grad_discrimins = T.grad(cost_discr, discrimins)
grad_bias_discrimins = T.grad(cost_discr, bias_discrimins)

autoencoder = SimpleDescent.AdaGrad(param_enc + [encodes, bias_encodes] + 
			param_gen + [initiates, bias_initiates], 
			grad_enc + [grad_encodes, grad_bias_encodes] + 
			grad_gen_enc + [grad_initiates_enc, grad_bias_initiates_enc])

train_autoencoder = autoencoder.step([x_enc], [x_gen_enc, z_enc, cost_enc], 0.05)

descent = SimpleDescent.CenterReg(param_gen + [initiates, bias_initiates] + 
				param_discr + [discrimins, bias_discrimins], 
			grad_gen + [grad_initiates, grad_bias_initiates]+ 
				grad_discr + [grad_discrimins, grad_bias_discrimins])


train = descent.step([x_in, z], [x, discr_in, discr_gen, cost_discr, cost_gen], 0.001)

descent_discr = SimpleDescent.AdaGrad(param_discr + [discrimins, bias_discrimins], 
			grad_discr + [grad_discrimins, grad_bias_discrimins])
train_discr = descent_discr.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.001)

generate = theano.function([x_in, z], [x, discr_gen])

################################# training ##########################
auto_steps = 000
auto_cost = numpy.zeros(auto_steps)

print "AUTOENCODE"
for t in range(auto_steps):
	xx, zz = makeSample(batch_size)
	x_enc, z_enc, cost_enc = train_autoencoder(xx)
	
	print cost_enc
	auto_cost[t] = cost_enc


train_steps = 000
train_dist = numpy.zeros(train_steps)
train_avg = numpy.zeros([train_steps, 2])
train_cost = numpy.zeros([train_steps, 2])

print "TRAIN"
for t in range(train_steps):
	
	xx, zz = makeSample(batch_size)
	train_discr(xx, zz)
	
	xx, zz = makeSample(batch_size)
	
	x_out, discr_in, discr_gen, cost_discr, cost_gen = train(xx, zz)
	
	train_avg[t, 0] = discr_in.mean()
	train_avg[t, 1] = discr_gen.mean()
	print (discr_in.mean(), discr_gen.mean())
	
	train_cost[t, 0] = cost_discr
	train_cost[t, 1] = cost_gen
	print (cost_discr, cost_gen)
	
	dist = energyDst(numpy.reshape(xx, [batch_size, -1]), numpy.reshape(x_out[0:batch_size, :], [batch_size, -1]))
	train_dist[t] = dist
	print dist

print "DISCRIMINATOR"

for i in range(00):
	xx, zz = makeSample(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost

#################### SAVE INFO ########################
with open("info.p", "rb") as f:
	pickle_file = pickle.Unpickler(f)

	last_auto_cost = pickle_file.load()
	last_train_dist = pickle_file.load()
	last_train_avg = pickle_file.load()
	last_train_cost = pickle_file.load()

with open("info.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	
	pickle_file.dump(auto_cost)
	pickle_file.dump(train_dist)
	pickle_file.dump(train_avg)
	pickle_file.dump(train_cost)

##################3 SAVE DATA ###############################
with open("network.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	
	pickle_file.dump(initiates.get_value())
	pickle_file.dump(bias_initiates.get_value())
	pickle_file.dump(encodes.get_value())
	pickle_file.dump(bias_encodes.get_value())
	pickle_file.dump(discrimins.get_value())
	pickle_file.dump(bias_discrimins.get_value())
	generator.dump(pickle_file)
	encoder.dump(pickle_file)
	discriminator.dump(pickle_file)

#################3 PLOT #####################
import matplotlib.pyplot as P

xx, zz = makeSample(batch_size)



P.plot(numpy.log(train_dist))
P.plot(numpy.log(last_train_dist))

P.show()

