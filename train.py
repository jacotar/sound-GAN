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
discriminator = freq_GAN.Discriminator.load(path + "/discriminator.p")

sample_size = generator.size_from(1)

#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
sampler = sampling.AudioFileFreqSampler("but_one_day.wav", sample_size, 128, 20)
#sampler = sampling.SinusSampler(sample_size)

def make_random(batch_size):
	return numpy.random.randn(batch_size, generator.gen_dim)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size])

################## network ###################3
batch_size = 200

z = T.dmatrix('z')
#z0 = T.reshape(z, [-1, 1, batch_size]) ### !!!!!!!!!!!!!!!!
x_gen = generator(z)

x_in = T.dtensor3('x_in')

x_full = T.concatenate([x_gen, x_in], 0)
x_full.name = "x_full"
discr = discriminator(x_full)

discr_gen = discr[0:batch_size]
discr_in = discr[batch_size:2*batch_size]

cost_discr = T.log(discr_in).mean() + T.log(1 - discr_gen).mean()


cost_gen = 0

for i in range(discriminator.num):
	phi_full = discriminator.ys[i]
	phi_gen = phi_full[0:batch_size, :, :]
	phi_in = phi_full[batch_size:2*batch_size, :, :]
	
	
	mean_in = phi_in.mean([0, 1])
	mean_gen = phi_gen.mean([0, 1])
	mean = 0.5 * (mean_in + mean_gen)

	var_in = T.sqr(phi_in - mean_in).mean([0, 1])
	var_gen = T.sqr(phi_gen - mean_gen).mean([0, 1])
	var = 0.5 * (var_in + T.sqr(mean_in) + var_gen + T.sqr(mean_gen)) - T.sqr(mean)
	
	cost_gen += -T.sqr(mean_in - mean_gen).mean()
	cost_gen += -0.03 * T.sqr(T.log(var_in) - T.log(var_gen)).mean()

	cost_discr += -0.1 * T.sqr(mean).mean()
	cost_discr += -0.003 * T.sqr(T.log(var)).mean()


################# descent ##################333
param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen, 1.0)

param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr, 1.0)

descent = SimpleDescent.AdaGrad(param_gen + param_discr, grad_gen + grad_discr)

train = descent.step([x_in, z], [x_gen, discr_in, discr_gen, cost_discr, cost_gen], 0.01)


descent_discr = SimpleDescent.AdaGrad(param_discr, grad_discr)
train_discr = descent_discr.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.01)


###################### training #############

print "DISCRIMINATOR"

for i in range(20):
        xx, temp = sampler.batch(batch_size)
	zz = make_random(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost

print "TRAIN"

epochs = 20
epoch_size = 50
train_dist = numpy.zeros(epochs)
train_avg = numpy.zeros([epochs, 2])
train_cost = numpy.zeros([epochs, 2])
train_ranks = numpy.zeros([epochs, generator.num])


for e in range(epochs):
	print e
	for ee in range(epoch_size):
		t = e * epoch_size + ee
	
        	xx, temp = sampler.batch(batch_size)
		zz = make_random(batch_size)
	
		x_out, discr_in, discr_gen, cost_discr, cost_gen = train(xx, zz)
	
		print (discr_in.mean(), discr_gen.mean(), numpy.sum(generator.ranks())), 
		print "\r", 
	

	train_avg[e, 0] = discr_in.mean()
	train_avg[e, 1] = discr_gen.mean()
	print train_avg[e, :]
	
	train_cost[e, 0] = cost_discr
	train_cost[e, 1] = cost_gen
	print train_cost[e, :]

	train_ranks[e, :] = generator.ranks()
	print train_ranks[e, :]
	
	dist = sampling.energyDst(
		numpy.reshape(xx, [batch_size, -1]), 
		numpy.reshape(x_out[0:batch_size, :, :], [batch_size, -1]))
	train_dist[e] = dist
	print dist


print "DISCRIMINATOR"

for i in range(20):
        xx, temp = sampler.batch(batch_size)
	zz = make_random(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())

generator.save(path + "/generator.p")
discriminator.save(path + "/discriminator.p")

with open(path + "/info.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	
	pickle_file.dump(train_avg)
	pickle_file.dump(train_cost)
	pickle_file.dump(train_ranks)
	pickle_file.dump(train_dist)

