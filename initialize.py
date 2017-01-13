import freq_GAN
import NonLinearities as NL

import numpy
import theano.tensor as T

import sampling

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


d = 1
win = 15

generator = freq_GAN.Generator(
	[100, 200, 100, 1], 
	[2, 5, 6], 
	[NL.Identity(), NL.Identity(), NL.Identity()])
generator.init_random()

discriminator = freq_GAN.Discriminator(
	1, 
	nonlin_sigmoid,
	[1, 800, 1000], 
	[win, 4], 
	[NL.Maxs(4), NL.Maxs(4)])
discriminator.init_random()

'''

num = 20

generator = freq_GAN.Generator(
	[700, 2000, 1000, num*2], 
	[2, 4, 5], 
	[NL.MaxsZero(3), NL.Maxs(3), NL.Identity()])
generator.init_random()

discriminator = freq_GAN.Discriminator(
	1, 
	nonlin_sigmoid,
	[num*2, 1000, 1500], 
	[8, 5], 
	[NL.Maxs(4), NL.Maxs(5)])
discriminator.init_random()
'''
#encoder = generator.get_encoder(nonlin_tan, [NL.Maxs(4), NL.Maxs(3), NL.Maxs(2)])
#encoder = generator.get_encoder(NL.Identity(), [NL.Identity(), NL.Identity(), NL.Identity()])
#encoder.init_random()

sample_size = generator.size_from(1)
gen_dim = generator.gen_dim
print gen_dim
data_dim = generator.dim[generator.num]


win = 8
#filt = (numpy.random.rand(win, d) - 0.5) / numpy.sqrt(win)
filt = numpy.ones([win, d]) / win
sampler = sampling.GaussianProcess(sample_size, filt, 1.0)


#sampler = sampling.AudioFileSampler(["Zece/audio"+str(i+1).zfill(2)+".wav" for i in range(23)], sample_size, 1024, num)

generator.save(path + "/generator.p")
discriminator.save(path + "/discriminator.p")
#encoder.save(path + "/encoder.p")
sampler.save(path+"/sampler.p")

'''
########################## encoder
batch_size = gen_dim * 2
ff = numpy.reshape(sampler.toFreq(sampler.batch(batch_size)), [batch_size, -1])
avg_ff = numpy.mean(ff, axis=0)
print avg_ff.shape
U, s, V = numpy.linalg.svd(ff- avg_ff, full_matrices=False)

encode = numpy.swapaxes(V[0:gen_dim, :], 0, 1)*numpy.sqrt(batch_size) / (s[0:gen_dim])
encode_bias = -numpy.tensordot(encode, avg_ff, ([0], [0]))
encode = numpy.reshape(encode, 
		[sample_size, data_dim, gen_dim])

z_enc = numpy.tensordot(ff, numpy.reshape(encode, [-1, gen_dim]), ([1], [0])) + encode_bias
mean_enc = numpy.mean(z_enc, axis=0)
var_enc = numpy.mean(numpy.square(z_enc - mean_enc), axis=0)
print numpy.mean(mean_enc)
print numpy.mean(var_enc)

g_num = generator.num
V = numpy.reshape(V[0:generator.dim[g_num-1], :], 
	[generator.dim[g_num-1], sample_size, -1])
generator.filters[g_num-1].set_value(
	V * numpy.reshape(s[0:generator.dim[g_num-1]], [-1, 1, 1])
	)

##for i in range(g_num-1):
##	generator.filters[i].set_value(
##		generator.filters[i].get_value()
##		)

import pickle
with open(path + "/linear_encoder.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	pickle_file.dump(encode)
	pickle_file.dump(encode_bias)

import matplotlib.pyplot as P

P.subplot(221)
P.plot(s)

P.subplot(222)
P.plot(mean_enc)
P.plot(var_enc)

P.subplot(223)
P.imshow(encode[:, :, 0])

P.subplot(224)
P.imshow(encode[:, :, 1])

P.show()

'''
