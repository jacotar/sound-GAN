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

d = 3
win = 15

generator = freq_GAN.Generator(
	100, 
	[nonlin_relu, nonlin_relu],
	[700, 400, 1], 
	[4, win], 
	[nonlin_relu, NL.Identity()])
generator.init_random()

discriminator = freq_GAN.Discriminator(
	1, 
	nonlin_sigmoid,
	[1, 300, 600], 
	[win, 4], 
	[NL.Maxs(4), NL.Maxs(5)])
discriminator.init_random()

encoder = generator.get_encoder(nonlin_tan, [NL.Maxs(4), NL.Maxs(3)])
encoder.init_random()
'''

generator = freq_GAN.Generator(
	60, 
	[NL.Identity(), NL.Identity()],
	[200, 100, 1], 
	[4, win], 
	[NL.Identity(), NL.Identity()])
generator.init_random()

discriminator = freq_GAN.Discriminator(
	1, 
	nonlin_sigmoid,
	[1, 150, 200], 
	[win, 4], 
	[NL.Maxs(4), NL.Maxs(5)])
discriminator.init_random()

encoder = generator.get_encoder(nonlin_tan, [NL.Identity(), NL.Identity()])
encoder.init_random()
'''
import numpy
import pickle
filt = (numpy.random.rand(8, d) - 0.5) / numpy.sqrt(win)
with open(path+"/gaussian_process.p", "wb") as f:
	pick = pickle.Pickler(f)
	pick.dump(filt)


generator.save(path + "/generator.p")
discriminator.save(path + "/discriminator.p")
encoder.save(path + "/encoder.p")
