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

read = False

try:
	os.mkdir(path)
except OSError:
	print "directory exists"
	read = True


nonlin_tan = NL.Function(T.arctan)
nonlin_sigmoid = NL.Function(T.nnet.nnet.sigmoid)
nonlin_relu = NL.Function(T.nnet.nnet.relu)

if read:
	generator = freq_GAN.Generator.load(path + "/generator.p")
	discriminator = freq_GAN.Discriminator.load(path + "/discriminator.p")
else:
	generator = freq_GAN.Generator(
		100,
		[nonlin_relu],
		[800, 1], 
		[30], 
		[NL.Identity()])
	generator.init_random()
	
	discriminator = freq_GAN.Discriminator(
		[1, 60, 200], 
		[6, 5], 
		[NL.Maxs(4), NL.Maxs(5)])
	discriminator.init_random()

#############################
sample_size = generator.size_from(1)

#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
sampler = sampling.SinusSampler(sample_size)

def make_random(batch_size):
	return numpy.random.randn(generator.gen_dim, batch_size)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size])

################## network ###################3
batch_size = 500

z = T.dmatrix('z')
#z0 = T.reshape(z, [-1, 1, batch_size]) ### !!!!!!!!!!!!!!!!
x_gen = generator(z)

x_in = T.dtensor3('x_in')

x_full = T.concatenate([x_gen, x_in], 2)
discr = discriminator(x_full)

discr_gen = discr[0:batch_size]
discr_in = discr[batch_size:2*batch_size]

cost_discr = T.log(discr_in).mean() + T.log(1 - discr_gen).mean()


#phi_full = T.reshape(discriminator.ys[discriminator.num-2], [-1, batch_size])
cost_gen = 0

for i in range(discriminator.num):
	phi_full = discriminator.ys[i]
	phi_gen = phi_full[:, :, 0:batch_size]
	phi_in = phi_full[:, :, batch_size:2*batch_size]
	
	#cost_gen = -energyDistTheano(phi_gen, phi_in)
	
	
	mean_in = phi_in.mean([1, 2], keepdims=True)
	mean_gen = phi_gen.mean([1, 2], keepdims=True)
	mean = 0.5 * (mean_in + mean_gen)

	var_in = T.sqr(phi_in - mean_in).mean([1, 2])
	var_gen = T.sqr(phi_gen - mean_gen).mean([1, 2])
	var = 0.5 * (var_in + T.sqr(mean_in.mean([1, 2])) + var_gen + T.sqr(mean_gen.mean([1, 2]))) - T.sqr(mean.mean([1, 2]))
	
	cost_gen += -T.sqr(mean_in - mean_gen).mean([0, 1, 2])
	cost_gen += -0.03 * T.sqr(T.log(var_in) - T.log(var_gen)).mean()

	cost_discr += -0.1 * T.sqr(mean).mean([0, 1, 2])
	cost_discr += -0.003 * T.sqr(T.log(var)).mean()


################# descent ##################333
param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen, 1.0)

param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr, 1.0)

descent = SimpleDescent.CenterReg(param_gen + param_discr, grad_gen + grad_discr)

train = descent.step([x_in, z], [x_gen, discr_in, discr_gen, cost_discr, cost_gen], 0.01)


descent_discr = SimpleDescent.AdaGrad(param_discr, grad_discr)
train_discr = descent_discr.step([x_in, z], [discr_in, discr_gen, cost_discr], 0.01)


###################### training #############

print "DISCRIMINATOR"

for i in range(20):
        xx = sampler.batch(batch_size)
	zz = make_random(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz)
	print (discr_in.mean(), discr_gen.mean())
	print cost

print "TRAIN"

epochs = 2000
epoch_size = 50
train_dist = numpy.zeros(epochs)
train_avg = numpy.zeros([epochs, 2])
train_cost = numpy.zeros([epochs, 2])
train_ranks = numpy.zeros([epochs, generator.num])

pickle_file = pickle.Pickler(open(path + "/generator_history.p", "wb"))

for e in range(epochs):
	print e
	for ee in range(epoch_size):
		t = e * epoch_size + ee
	
        	xx = sampler.batch(batch_size)
		zz = make_random(batch_size)
	
		x_out, discr_in, discr_gen, cost_discr, cost_gen = train(xx, zz)
	
		print (discr_in.mean(), discr_gen.mean(), numpy.sum(generator.ranks())), 
		print "\r", 
	
	generator.writePickle(pickle_file)

	train_avg[e, 0] = discr_in.mean()
	train_avg[e, 1] = discr_gen.mean()
	print train_avg[e, :]
	
	train_cost[e, 0] = cost_discr
	train_cost[e, 1] = cost_gen
	print train_cost[e, :]

	train_ranks[e, :] = generator.ranks()
	print train_ranks[e, :]
	
	dist = sampling.energyDst(
		numpy.reshape(xx, [-1, batch_size]), 
		numpy.reshape(x_out[:, :, 0:batch_size], [-1, batch_size]))
	train_dist[e] = dist
	print dist


print "DISCRIMINATOR"

for i in range(20):
        xx = sampler.batch(batch_size)
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

