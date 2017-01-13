import freq_GAN
import sampling
import NonLinearities as NL
import SimpleDescent

import numpy
import theano.tensor as T

import pickle
import sys
import os

path = sys.argv[1]

generator = freq_GAN.Generator.load(path + "/generator.p")

sample_size = generator.size_from(1)

sampler = sampling.AudioFileSampler.load(path+"/sampler.p")

def make_random(batch_size):
	return numpy.random.randn(batch_size, generator.gen_dim)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size])

################## network ###################3

z = T.dmatrix('z')
batch_size = z.shape[0]

x_gen = generator(T.reshape(z, [-1, 1, generator.gen_dim])) ### !!!!!!!!!!!!!!!!

x_in = T.dtensor3('x_in')

cost_gen = -sampling.energyDstTheano2(
		T.reshape(x_gen, [batch_size, 1, -1]), 
		T.reshape(x_in, [batch_size, 1, -1]))

################# descent ##################333
param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen, 1.0)

descent = SimpleDescent.Grad(param_gen, grad_gen)

train = descent.step([x_in, z], [x_gen, cost_gen], 0.01)

###################### training #############

batch_size = 50

print "TRAIN"

epochs = 200
train_dist = numpy.zeros(epochs)


for t in range(epochs):	
        xx = sampler.toFreq(sampler.batch(batch_size))
	zz = make_random(batch_size)
	
	x_out, cost_gen = train(xx, zz)
	
	print cost_gen
	
	train_dist[t] = cost_gen


generator.save(path + "/generator.p")

with open(path + "/info.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	
	pickle_file.dump(train_dist)

