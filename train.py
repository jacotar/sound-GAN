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

#sampler = sampling.AudioFileSampler.load(path+"/sampler.p")
sampler = sampling.fromPickle(path+"/sampler.p")
#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
#sampler = sampling.AudioFileFreqSampler("but_one_day.wav", sample_size, 128, 20)
#sampler = sampling.SinusSampler(sample_size)
#sampler = sampling.AudioFileSampler(["Zece/audio"+str(i+1).zfill(2)+".wav" for i in range(23)], sample_size)
'''
with open(path+"/gaussian_process.p", "rb") as f:
	pick = pickle.Unpickler(f)
	sampler = sampling.GaussianProcess(sample_size, pick.load(), 1.0)
'''

def make_random(batch_size):
	return numpy.random.randn(batch_size, generator.gen_dim)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size])

################## network ###################3

z = T.dmatrix('z')
batch_size = z.shape[0]

x_gen = generator(T.reshape(z, [-1, 1, generator.gen_dim])) ### !!!!!!!!!!!!!!!!

x_in = T.dtensor3('x_in')

x_full = T.concatenate([x_gen, x_in], 0)
discr = discriminator(x_full)

discr_gen = discr[0:batch_size]
discr_in = discr[batch_size:2*batch_size]

cost_discr = T.log(discr_in + 0.001).mean() + T.log(1.001 - discr_gen).mean()


cost_gen = 0

for i in range(discriminator.num):
	phi_full = discriminator.ys[i]
	phi_gen = phi_full[0:batch_size, :, :]
	phi_in = phi_full[batch_size:2*batch_size, :, :]

	mean_in = phi_in.mean([0, 1])
	mean_gen = phi_gen.mean([0, 1])
	mean = 0.5 * (mean_in + mean_gen)
	var = T.sqr(phi_full - mean).mean([0, 1])

	var_in = T.sqr(phi_in - mean_in).mean([0, 1])
	var_gen = T.sqr(phi_gen - mean_gen).mean([0, 1])
	#var = 0.5 * (var_in + T.sqr(mean_in) + var_gen + T.sqr(mean_gen)) - T.sqr(mean)
	#var = abs(var)
	
	cost_gen += -(mean_in - mean_gen).norm(2)
	cost_gen += -0.3 * (T.log(var_in + 0.001) - T.log(var_gen + 0.001)).norm(2)
	#cost_gen += -0.1 * generator.normL1()

	cost_discr += -0.001 * (mean).norm(2)
	cost_discr += -0.001 * (T.log(var + 0.001)).norm(2)


################# descent ##################333
time = T.dscalar('t')
decay = 1.0 + 0.0 * T.exp(-time / 200.0)

param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_gen*0.1, decay)

param_discr = discriminator.getParameters()
grad_discr = discriminator.getGradients(cost_discr*0.01, decay)

descent = SimpleDescent.Grad(param_gen + param_discr, grad_gen + grad_discr)

train = descent.step([x_in, z, time], [x_gen, discr_in, discr_gen, cost_discr, cost_gen], 0.001)


descent_discr = SimpleDescent.Grad(param_discr, grad_discr)
train_discr = descent_discr.step([x_in, z, time], [discr_in, discr_gen, cost_discr], 0.001)


###################### training #############

batch_size = 25

print "DISCRIMINATOR"

for i in range(100):
        xx = sampler.toFreq(sampler.batch(batch_size))
	zz = make_random(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz, i)
	print (discr_in.mean(), discr_gen.mean())
	print cost

print "TRAIN"

epochs = 600
epoch_size = 1 #16
train_dist = numpy.zeros(epochs)
train_avg = numpy.zeros([epochs, 2])
train_cost = numpy.zeros([epochs, 2])
train_percentiles = numpy.zeros([epochs, 2, 5])


for e in range(epochs):
	print e
	for ee in range(epoch_size):
		t = e * epoch_size + ee
	
        	xx = sampler.toFreq(sampler.batch(batch_size))
		zz = make_random(batch_size)
	
		x_out, discr_in, discr_gen, cost_discr, cost_gen = train(xx, zz, e)
	
		print (discr_in.mean(), discr_gen.mean()) 
	

	train_avg[e, 0] = discr_in.mean()
	train_avg[e, 1] = discr_gen.mean()
	print train_avg[e, :]
	
	train_cost[e, 0] = cost_discr
	train_cost[e, 1] = cost_gen
	print train_cost[e, :]
	
	train_percentiles[e, 0, :] = numpy.percentile(discr_in, [5.0, 25.0, 50.0, 75.0, 95.0])
	train_percentiles[e, 1, :] = numpy.percentile(discr_gen, [5.0, 25.0, 50.0, 75.0, 95.0])
	print train_percentiles[e, :, :]
	
	
	dist = sampling.energyDst(
		numpy.reshape(xx, [batch_size, -1]), 
		numpy.reshape(x_out[0:batch_size, :, :], [batch_size, -1]))
	train_dist[e] = dist
	print dist


print "DISCRIMINATOR"

for i in range(20):
        xx = sampler.toFreq(sampler.batch(batch_size))
	zz = make_random(batch_size)
	
	discr_in, discr_gen, cost = train_discr(xx, zz, i)
	print (discr_in.mean(), discr_gen.mean())

generator.save(path + "/generator.p")
discriminator.save(path + "/discriminator.p")

with open(path + "/info.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	
	pickle_file.dump(train_avg)
	pickle_file.dump(train_cost)
	pickle_file.dump(0)
	pickle_file.dump(train_percentiles)
	pickle_file.dump(train_dist)

