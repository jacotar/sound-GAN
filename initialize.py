import freq_GAN
import NonLinearities as NL

import theano.tensor as T

import sys
import os

path = sys.argv[1]

try:
	os.mkdir(path)
except OSError:
	pass

nonlin_tan = NL.Function(T.arctan)
nonlin_sigmoid = NL.Function(T.nnet.nnet.sigmoid)
nonlin_relu = NL.Function(T.nnet.nnet.relu)

generator = freq_GAN.Generator(
	100, 
	[nonlin_relu, nonlin_relu],
	[300, 200, 40], 
	[3, 5], 
	[NL.Maxs(3), NL.Identity()])
generator.init_random()

discriminator = freq_GAN.Discriminator(
	1, 
	nonlin_sigmoid,
	[40, 150, 300], 
	[5, 3], 
	[NL.Maxs(4), NL.Maxs(5)])
discriminator.init_random()

encoder = generator.get_encoder(nonlin_tan, [NL.Maxs(4), NL.Maxs(3)])
encoder.init_random()

generator.save(path + "/generator.p")
discriminator.save(path + "/discriminator.p")
encoder.save(path + "/encoder.p")
