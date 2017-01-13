import freq_GAN
import sampling

import numpy
import theano
import theano.tensor as T

import sys

import scipy.io.wavfile as W

path = sys.argv[1]

generator = freq_GAN.Generator.load(path + "/generator.p")

sample_size = generator.size_from(1)
sampler = sampling.AudioFileSampler.load(path+"/sampler.p")
#sampler = sampling.AudioFileSampler(["Zece/audio"+str(i+1).zfill(2)+".wav" for i in range(23)], sample_size)

z = T.dmatrix('z')

#x = generator(z)
x = generator(T.reshape(z, [-1, 1, generator.gen_dim])) ### !!!!!!!!!!!!!!!!

generate = theano.function([z], [x])

batch_size = int(sys.argv[2])
zz = numpy.random.randn(batch_size, generator.gen_dim)

ff_gen, = generate(zz)
xx_gen = sampler.fromFreq(ff_gen)

#xx_gen = sampler.fromFreq(sampler.toFreq(sampler.batch(batch_size)))

for i in range(batch_size):
	W.write("samples/audio"+str(i+1).zfill(5)+".wav", 44100, 
		numpy.reshape(xx_gen[i, :, :], [-1]))

