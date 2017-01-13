import freq_GAN
import sampling

import numpy
import theano
import theano.tensor as T

import sys

import matplotlib.pyplot as P

path = sys.argv[1]


generator = freq_GAN.Generator.load(path + "/generator.p")

sample_size = generator.size_from(1)

sampler = sampling.fromPickle(path+"/sampler.p")


batch_size = 4

z = T.dmatrix('z')

#x = generator(z)
x = generator(T.reshape(z, [-1, 1, generator.gen_dim])) ### !!!!!!!!!!!!!!!!

generate = theano.function([z], [x])

zz = numpy.random.randn(batch_size, generator.gen_dim)
ff_gen, = generate(zz)

ff_in = sampler.toFreq(sampler.batch(batch_size*100))

xx_in = sampler.batch(batch_size)
ids = sampling.nearest(ff_gen, ff_in)
ff_in = ff_in[ids, :, :]


P.figure(1)

print numpy.percentile(ff_in, [5, 25, 50, 75, 95])
print numpy.percentile(ff_gen, [5, 25, 50, 75, 95])

'''
P.subplot(221)
P.imshow(ff_in[0, :, :])

P.subplot(222)
P.imshow(ff_in[1, :, :])

P.subplot(223)
P.imshow(ff_in[2, :, :])

P.subplot(224)
P.imshow(ff_in[3, :, :])
'''
P.subplot(221)
P.plot(ff_in[0, :, 0])
P.plot(ff_gen[0, :, 0])

P.subplot(222)
P.plot(ff_in[1, :, 0])
P.plot(ff_gen[1, :, 0])

P.subplot(223)
P.plot(ff_in[2, :, 0])
P.plot(ff_gen[2, :, 0])

P.subplot(224)
P.plot(ff_in[3, :, 0])
P.plot(ff_gen[3, :, 0])

P.show()

