import freq_GAN
import sampling

import SimpleDescent

import numpy
import theano
import theano.tensor as T

import pickle
import sys

path = sys.argv[1]

generator = freq_GAN.Generator.load(path + "/generator.p")
#encoder = freq_GAN.Discriminator.load(path + "/encoder.p")

sample_size = generator.size_from(1)

gen_dim = generator.gen_dim
data_dim = generator.dim[generator.num]
'''
encode = theano.shared(numpy.random.normal(0, 1.0/numpy.sqrt(sample_size*data_dim), 
		[sample_size, data_dim, gen_dim]))
encode_bias = theano.shared(numpy.zeros(gen_dim))
'''


with open(path + "/linear_encoder.p", "rb") as f:
	pickle_file = pickle.Unpickler(f)
	encode = theano.shared(pickle_file.load())
	encode_bias = theano.shared(pickle_file.load())
	#encode_bias = theano.shared(numpy.zeros(gen_dim))

sampler = sampling.AudioFileSampler.load(path+"/sampler.p")
#sampler = sampling.AudioFileSampler(["Zece/audio"+str(i+1).zfill(2)+".wav" for i in range(23)], sample_size)
#sampler = sampling.AudioFileFreqSampler("but_one_day.wav", sample_size, 128, 20)
#sampler = sampling.SinusSampler(sample_size)

'''import pickle
with open(path+"/gaussian_process.p", "rb") as f:
	pick = pickle.Unpickler(f)
	sampler = sampling.GaussianProcess(sample_size, pick.load(), 1.0)
'''
######################

x = T.dtensor3('x')
batch_size = x.shape[0]

z = T.tensordot(x, encode, ([1, 2], [0, 1])) + encode_bias
#encoder(x)

xx = generator(T.reshape(z, [-1, 1, generator.gen_dim]))

mean_enc = z.mean()
var_enc = T.sqr(z - mean_enc).mean()


cost_enc = -(xx - x).norm(2, axis=[1, 2]).mean() / (sample_size * data_dim)
#cost_enc = -T.sqr(xx - x).mean(axis=[0, 1, 2])
#cost_enc += -(mean_enc).norm(2)*0.01 - (T.log(var_enc)).norm(2)*0.001
#cost_enc += -0.01 * generator.normL1()

cost_enc *= 100

######################
time = T.dscalar('t')

param_enc = [encode, encode_bias]
grad_enc = [T.grad(cost_enc, encode), 
	    T.grad(cost_enc, encode_bias)]

param_gen = generator.getParameters()
grad_gen = generator.getGradients(cost_enc) 

descent = SimpleDescent.Momentum(
		param_gen, # + param_enc, 
		grad_gen) # + grad_enc)

train = descent.step([x], [xx, cost_enc], 0.1)

###################

batch_size = 10
steps = 1000
xxx = numpy.zeros([batch_size, sample_size, data_dim])

for t in range(steps):
        xx = sampler.toFreq(sampler.batch(batch_size))
	
	print sampling.energyDst(
		numpy.reshape(xx, [batch_size, -1]), 
		numpy.reshape(xxx, [batch_size, -1]), 
	)
	
        xxx, cost = train(xx)
	if t % 5 ==0:
		print numpy.percentile(xx, [5, 25, 50, 75, 95])
		print numpy.percentile(xxx, [5, 25, 50, 75, 95])

        print cost



generator.save(path+"/generator.p")
#encoder.save(path+"/encoder.p")


with open(path + "/linear_encoder.p", "wb") as f:
	pickle_file = pickle.Pickler(f)
	pickle_file.dump(encode.get_value())
	pickle_file.dump(encode_bias.get_value())


import matplotlib.pyplot as P

batch_size = 100

ff_in = sampler.toFreq(sampler.batch(batch_size))

ff_gen, cost = train(ff_in)
#xx_gen = numpy.reshape(sampler.fromFreq(ff_gen), [batch_size, sampler.sample_size * sampler.window_size])
#xx_in = numpy.reshape(xx_in, [batch_size, sampler.sample_size * sampler.window_size])

print numpy.percentile(ff_in, [5, 25, 50, 75, 95])
print numpy.percentile(ff_gen, [5, 25, 50, 75, 95])

P.figure(1)

P.subplot(221)
P.imshow(ff_in[0, :, :])

P.subplot(222)
P.imshow(ff_gen[0, :, :])

P.subplot(223)
P.imshow(ff_in[1, :, :])

P.subplot(224)
P.imshow(ff_gen[1, :, :])

P.show()

ff_in = sampler.toFreq(sampler.batch(batch_size))
print sampling.energyDst(
	numpy.reshape(ff_in, [batch_size, -1]), 
	numpy.reshape(ff_gen, [batch_size, -1]), 
	)

