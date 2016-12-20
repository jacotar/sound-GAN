import numpy
import theano
import theano.tensor as T

import freq_GAN

import pickle

import sys

import matplotlib.pyplot as P

path = sys.argv[1]
'''
generator = freq_GAN.Generator.load(path+"/generator.p")

f = open(path+"/generator_history.p", "rw")
pickle_file = pickle.Unpickler(f)

pickle_file.load()
pickle_file.load()
pickle_file.load()
pickle_file.load()
pickle_file.load()

d_filters = []
d_biases = []
d_direct_filters = []
d_direct_biases = []

for i in range(generator.num):
	d_filters.append(pickle_file.load() - generator.filters[i].get_value())
	d_biases.append(pickle_file.load() - generator.biases[i].get_value())
	d_direct_filters.append(pickle_file.load() - generator.direct_filters[i].get_value())
	d_direct_biases.append(pickle_file.load() - generator.direct_biases[i].get_value())

dists1 = []
dists2 = []

for i in range(1000):
	dist1 = numpy.zeros(1)
	dist2 = numpy.zeros(1)
	try:
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		pickle_file.load()
		
		for i in range(generator.num):
			filters = pickle_file.load()
			biases = pickle_file.load()
			direct_filters = pickle_file.load()
			direct_biases = pickle_file.load()
			
			dist2 += numpy.sum(numpy.square(filters - generator.filters[i].get_value()), axis=(0, 1, 2))
			dist2 += numpy.sum(numpy.square(biases - generator.biases[i].get_value()))
			dist2 += numpy.sum(numpy.square(direct_filters - generator.direct_filters[i].get_value()), axis=(0, 1))
			dist2 += numpy.sum(numpy.square(direct_biases - generator.direct_biases[i].get_value()))
			
			dist1 += numpy.tensordot(d_filters[i], filters, [[0, 1, 2], [0, 1, 2]])
			dist1 += numpy.tensordot(d_biases[i], biases, [[0], [0]])
			dist1 += numpy.tensordot(d_direct_filters[i], direct_filters, [[0, 1], [0, 1]])
			dist1 += numpy.tensordot(d_direct_biases[i], direct_biases, [[0], [0]])
	except (EOFError, pickle.UnpicklingError):
		break
	
	dists1.append(dist1)
	dists2.append(dist2)

P.plot(dists1)
P.plot(dists2)
P.show()

'''
f = open(path+"/generator_history.p", "rw")
pickle_file = pickle.Unpickler(f)

generator = freq_GAN.Generator(pickle_file.load(),
		pickle_file.load(),
		pickle_file.load(),
		pickle_file.load(),
		pickle_file.load())

generator.filters = []
generator.biases = []
generator.direct_filters = []
generator.direct_biases = []

for i in range(generator.num):
	generator.filters.append(theano.shared(pickle_file.load()))
	generator.biases.append(theano.shared(pickle_file.load()))
	generator.direct_filters.append(theano.shared(pickle_file.load()))
	generator.direct_biases.append(theano.shared(pickle_file.load()))



batch_size = 1

z = T.dmatrix('z')
# z0 = T.reshape(z, [-1, 1, batch_size]) ### !!!!!!!!!!!!!!!!
x = generator(z)

generate = theano.function([z], [x])


zz = numpy.random.randn(generator.gen_dim, batch_size)

for i in range(1000):
	print i
	try:
		generator.fromPickle(pickle_file)
		generator.fromPickle(pickle_file)
		generator.fromPickle(pickle_file)
		generator.fromPickle(pickle_file)
		generator.fromPickle(pickle_file)
	except (EOFError, pickle.UnpicklingError):
		break
	xx, = generate(zz)
	P.plot(xx[0, :, 0])

P.show()

f.close()
