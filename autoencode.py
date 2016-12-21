import freq_GAN
import sampling

import SimpleDescent

import numpy
import theano
import theano.tensor as T

import sys

path = sys.argv[1]

generator = freq_GAN.Generator.load(path + "/generator.p")
encoder = freq_GAN.Discriminator.load(path + "/encoder.p")

sample_size = generator.size_from(1)

#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
sampler = sampling.AudioFileFreqSampler("but_one_day.wav", sample_size, 128, 20)
#sampler = sampling.SinusSampler(sample_size)

######################

x = T.dtensor3('x')

z = encoder(x)
z.name = "z"

xx = generator(z)

mean_enc = z.mean(0)
var_enc = T.sqr(z - mean_enc).mean([0, 1])

cost_enc = -T.sqr(xx - x).mean([0, 1, 2])
cost_enc += -T.sqr(mean_enc).mean()*0.1 - T.sqr(T.log(var_enc)).mean()*0.001


######################
time = T.dscalar('t')

param_gen = generator.getParameters() + encoder.getParameters()
grad_gen = generator.getGradients(cost_enc, T.exp(-time / 500)) + encoder.getGradients(cost_enc, T.exp(-time / 500)) 


descent = SimpleDescent.AdaGrad(param_gen, grad_gen)

train = descent.step([x, time], [xx, cost_enc], 0.1)

###################

batch_size = 100
steps = 600
for t in range(steps):
        xx, temp = sampler.batch(batch_size)
        xxx, cost = train(xx, t)

        print cost



generator.save(path+"/generator.p")
encoder.save(path+"/encoder.p")


import matplotlib.pyplot as P

batch_size = 5
ff_in, xx_in = sampler.batch(batch_size)

ff_gen, cost = train(ff_in, 0)
xx_gen = sampler.inverse_transform(ff_gen)



P.figure(1)

P.subplot(211)
for i in range(batch_size):
        P.plot(xx_in[i, :, 0])

P.subplot(212)
for i in range(batch_size):
        P.plot(xx_gen[i, :])

P.show()

