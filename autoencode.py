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
#sampler = sampling.AudioFileFreqSampler("but_one_day.wav", sample_size, 128, 20)
#sampler = sampling.SinusSampler(sample_size)
import pickle
with open(path+"/gaussian_process.p", "rb") as f:
	pick = pickle.Unpickler(f)
	sampler = sampling.GaussianProcess(sample_size, pick.load(), 0.2)

######################

x = T.dtensor3('x')

z = encoder(x)
z.name = "z"

xx = generator(z)

mean_enc = z.mean(0)
var_enc = T.sqr(z - mean_enc).mean([0, 1])

cost_enc = -T.norm(xx - x, axis=[0, 1, 2])
cost_enc += -T.norm(mean_enc)*0.1 - T.norm(T.log(var_enc))*0.001
cost_enc += -0.1 * generator.normL1()


######################
time = T.dscalar('t')

param_gen = encoder.getParameters() + generator.getParameters()
grad_gen = encoder.getGradients(cost_enc, T.exp(-time / 500)) + generator.getGradients(cost_enc, T.exp(-time / 500)) 


descent = SimpleDescent.AdaGrad(param_gen, grad_gen)

train = descent.step([x, time], [xx, cost_enc], 0.02)

###################

batch_size = 100
steps = 300
for t in range(steps):
        xx = sampler.batch(batch_size)
        xxx, cost = train(xx, t)

        print cost



generator.save(path+"/generator.p")
encoder.save(path+"/encoder.p")


import matplotlib.pyplot as P

batch_size = 5

xx_in = sampler.batch(batch_size)

xx_gen, cost = train(xx_in, 0)
#xx_gen = sampler.inverse_transform(ff_gen)




P.figure(1)

P.subplot(211)
for i in range(batch_size):
        P.plot(xx_in[i, :, 0])

P.subplot(212)
for i in range(batch_size):
        P.plot(xx_gen[i, :, 0])

P.show()

