import freq_GAN
import sampling
import NonLinearities as NL

import numpy
import theano
import theano.tensor as T

import sys

import matplotlib.pyplot as P

import pickle

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
	generator = freq_GAN.Generator.load(sys.argv[1])
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
sampler = sampling.SinusSampler(sample_size)



batch_size = 5

z = T.dmatrix('z')
# z0 = T.reshape(z, [-1, 1, batch_size]) ### !!!!!!!!!!!!!!!!
x = generator(z)

generate = theano.function([z], [x])

# zz = numpy.random.randn(generator.dim[0], batch_size)
zz = numpy.random.randn(generator.gen_dim, batch_size)# * numpy.random.randint(0, 2, [generator.dim[0], batch_size*10])
xx_gen, = generate(zz)

xx_in = sampler.batch(batch_size*100)

#xx_in = sampler.batch(batch_size)
ids = sampling.nearest(xx_gen, xx_in)
xx_in = xx_in[:, :, ids]


P.figure(1)

P.subplot(211)
for i in range(batch_size):
        P.plot(xx_in[0, :, i])

P.subplot(212)
for i in range(batch_size):
        P.plot(xx_gen[0, :, i])

P.show()

