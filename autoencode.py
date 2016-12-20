import freq_GAN
import sampling

import NonLinearities as NL

import SimpleDescent

import numpy
import theano
import theano.tensor as T

import sys


nonlin_tan = NL.Function(T.arctan)
nonlin_sigmoid = NL.Function(T.nnet.nnet.sigmoid)
nonlin_realu = NL.Function(T.nnet.nnet.relu)

if len(sys.argv) <= 1:
	generator = freq_GAN.Generator(
		[100, 200, 100, 1], 
		[2, 5, 10], 
		[NL.Maxs(3), NL.Maxs(2), NL.Identity()])
	generator.init_random()
else:
	generator = freq_GAN.Generator.load(sys.argv[1])

generator.init_autoencoder()

#############################
sample_size = generator.size_from(1)

#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
sampler = sampling.SinusSampler(sample_size)

######################

x = T.dtensor3('x')

xx = generator.apply_autoencoder(x)

mean_enc = generator.z_enc.mean([1, 2], keepdims=True)
var_enc = T.sqr(generator.z_enc - mean_enc).mean([0, 1, 2])

cost_enc = -T.sqr(xx - x).mean([0, 1, 2])
cost_enc += -T.sqr(mean_enc).mean([0, 1, 2])*0.1 - T.sqr(var_enc - 1.0).mean()*0.1


######################
time = T.dscalar('t')

param_gen = generator.getAutoencoderParameters()
grad_gen = generator.getAutoencoderGradients(cost_enc, T.exp(-time / 500))


descent = SimpleDescent.AdaGrad(param_gen, grad_gen)

train = descent.step([x, time], [xx, cost_enc], 0.1)

###################

batch_size = 300
steps = 500
for t in range(steps):
        xx = sampler.batch(batch_size)
        xxx, cost = train(xx, t)

        print cost



generator.save("SAVE/generator.p")

import matplotlib.pyplot as P

batch_size = 5
xx_in = sampler.batch(batch_size)

xx_gen, cost = train(xx_in, 0)

P.figure(1)

P.subplot(211)
for i in range(batch_size):
        P.plot(xx_in[0, :, i])

P.subplot(212)
for i in range(batch_size):
        P.plot(xx_gen[0, :, i])

P.show()

