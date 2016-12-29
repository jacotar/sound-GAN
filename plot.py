import freq_GAN
import sampling
import NonLinearities as NL

import numpy
import theano
import theano.tensor as T

import sys

import matplotlib.pyplot as P

import pickle

path = sys.argv[1]

nonlin_tan = NL.Function(T.arctan)
nonlin_sigmoid = NL.Function(T.nnet.nnet.sigmoid)
nonlin_relu = NL.Function(T.nnet.nnet.relu)

if len(sys.argv) <= 1:
	generator = freq_GAN.Generator(
		[3, 3, 2], 
		[4, 5], 
		[nonlin_tan, nonlin_tan])
	generator.init_random()

        generator.filters[0].set_value(generator.filters[0].get_value() * 0.0)
        generator.biases[0].set_value([1, 0, 0])
else:
	generator = freq_GAN.Generator.load(path + "/generator.p")
	'''f = open(sys.argv[1], "rb")
	pickle_file = pickle.Unpickler(f)
	for i in range(1074):
		print i
		try:
			generator.fromPickle(pickle_file)
		except (EOFError, pickle.UnpicklingError):
			break
	
	f.close()

generator.save("generator.p")
'''
sample_size = generator.size_from(1)
#sampler = sampling.AudioFileSampler("but_one_day.wav", sample_size)
#sampler = sampling.SinusSampler(sample_size)
with open(path+"/gaussian_process.p", "rb") as f:
	pick = pickle.Unpickler(f)
	sampler = sampling.GaussianProcess(sample_size, pick.load(), 0.2)


batch_size = 4

z = T.dmatrix('z')
# z0 = T.reshape(z, [-1, 1, batch_size]) ### !!!!!!!!!!!!!!!!
x = generator(z)

generate = theano.function([z], [x])

# zz = numpy.random.randn(generator.dim[0], batch_size)
zz = numpy.random.randn(batch_size, generator.gen_dim)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size*10])
xx_gen, = generate(zz)

xx_in = sampler.batch(batch_size*1000)

#xx_in = sampler.batch(batch_size)
ids = sampling.nearest(xx_gen, xx_in)
xx_in = xx_in[ids, :, :]


P.figure(1)
'''
P.subplot(211)
for i in range(batch_size):
        P.plot(xx_in[i, :, 0])

P.subplot(212)
for i in range(batch_size):
        P.plot(xx_gen[i, :, 0])
'''
P.subplot(211)
P.plot(xx_in[0, :, 0])
P.plot(xx_gen[0, :, 0])

P.subplot(221)
P.plot(xx_in[1, :, 0])
P.plot(xx_gen[1, :, 0])

P.subplot(212)
P.plot(xx_in[2, :, 0])
P.plot(xx_gen[2, :, 0])

P.subplot(222)
P.plot(xx_in[3, :, 0])
P.plot(xx_gen[3, :, 0])

P.show()

