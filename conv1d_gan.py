import numpy
import theano.tensor as T
import theano
import theano.typed_list

'''
num_conv = 5
dimensions = [1, 7, 50, 200, 200, 300]
window_sizes = [8, 8, 4, 4, 4]
max_mults = [2, 2, 2, 2, 3]

num_conv = 1
dimensions = [1, 124]
window_sizes = [2024]
max_mults = [2]
'''
num_conv = 2
dimensions = [1, 5, 10]
window_sizes = [16, 16]
max_mults = [2, 2]

rng = numpy.random
filters = [theano.shared((
		rng.rand(max_mults[i], dimensions[i+1], 
				dimensions[i], window_sizes[i])-0.5)/(window_sizes[i]*1.0))
	for i in range(num_conv)
	]
step_filters = [theano.shared(
		numpy.zeros([max_mults[i], dimensions[i+1], 
				dimensions[i], window_sizes[i]], numpy.float64))
	for i in range(num_conv)]

biases = [theano.shared((
	rng.rand(max_mults[i], dimensions[i+1])-0.5)/(window_sizes[i]*1.0))
	for i in range(num_conv)
	]
step_biases = [theano.shared(
	numpy.zeros([max_mults[i], dimensions[i+1]], numpy.float64))
	for i in range(num_conv)]



discrimins = theano.shared((rng.rand(dimensions[num_conv], 1)-0.5))
step_discrimins = theano.shared(numpy.zeros([dimensions[num_conv], 1], numpy.float64))

bias_discrimins = theano.shared(0.0)
step_bias_discrimins = theano.shared(0.0)


x = T.dmatrix('x')
sources = T.dcol('sources')
batch_size, num = T.shape(x)

y = x.dimshuffle(0, 'x', 'x', 1);
y = T.unbroadcast(y, 1);
y = T.unbroadcast(y, 2);

for i in range(num_conv):
	filt = T.reshape(filters[i], 
		[dimensions[i+1] * max_mults[i], dimensions[i],
		1, window_sizes[i]])
	bias = T.reshape(biases[i], [dimensions[i+1] * max_mults[i]]);
	bias = bias.dimshuffle('x', 0, 'x', 'x')
	y = T.nnet.conv.conv2d(y, filt, subsample=(window_sizes[i], 1)) + bias
	
	num = T.shape(y)[3]
	y = T.reshape(y, [batch_size, dimensions[i+1], max_mults[i], num])
	y = y.max(2, True)


discr = T.dot(
	T.reshape(y.dimshuffle(0, 3, 1), [batch_size*num, dimensions[num_conv]]),
	discrimins)
discr = T.nnet.sigmoid(T.reshape(discr, [batch_size, num]) + bias_discrimins)
cost = numpy.log(numpy.abs(sources - discr)).mean().mean()
discr = discr.mean(1)

grad_filters = T.grad(cost, filters)
grad_biases = T.grad(cost, biases)
grad_discrimins = T.grad(cost, discrimins)
grad_bias_discrimins = T.grad(cost, bias_discrimins)

train = theano.function([x, sources], [discr, cost], updates=
	[(step_filters[i], step_filters[i]*0.7 + grad_filters[i]) for i in range(num_conv)] + 
	[(step_biases[i], step_biases[i]*0.7 + grad_biases[i]) for i in range(num_conv)] + 
	[(step_discrimins, step_discrimins*0.7+ grad_discrimins)] + 
	[(step_bias_discrimins, step_bias_discrimins*0.7 + grad_bias_discrimins)]+

	[(filters[i], filters[i] + 0.002 * step_filters[i]) for i in range(num_conv)] + 
	[(biases[i], biases[i] + 0.002 * step_biases[i]) for i in range(num_conv)] + 
	[(discrimins, discrimins + 0.002 *step_discrimins)] + 
	[(bias_discrimins, bias_discrimins + 0.002 * step_bias_discrimins)]
	)
predict = theano.function([x], [discr])

# training on audio files
import wave

batch_size = 30
sample_size = numpy.product(window_sizes)

fichier_A = wave.open('Cymbals/Rev_Crsh.wav')
fichier_B = wave.open('Cymbals/Ride_01.wav')

max_A = fichier_A.getnframes() - sample_size
max_B = fichier_B.getnframes() - sample_size

source = numpy.zeros([batch_size*2, 1])
source[batch_size:batch_size*2] = 1
xx = numpy.zeros([batch_size*2, sample_size])

for i in range(200):
	print 'wave'
	for k in range(batch_size):
		fichier_A.setpos(numpy.random.randint(max_A))
		fichier_B.setpos(numpy.random.randint(max_B))
		xx[k, :] = numpy.fromstring(fichier_A.readframes(sample_size), numpy.int16) / 65536.0
		xx[batch_size + k, :] = numpy.fromstring(fichier_B.readframes(sample_size), numpy.int16) / 65536.0

	print 'train'
	discr, cost = train(xx, source)
	#print discr[0:batch_size].sum()
	#print discr[batch_size:batch_size*2].sum()
	print cost


# validation error
batch_size = 100
xx = numpy.zeros([batch_size, sample_size])

# set A
for k in range(batch_size):
		fichier_A.setpos(numpy.random.randint(max_A))
		xx[k, :] = numpy.fromstring(fichier_A.readframes(sample_size), numpy.int16) / 65536.0

discr = predict(xx)

print 'average A'
print numpy.average(discr)


# set B
for k in range(batch_size):
		fichier_B.setpos(numpy.random.randint(max_B))
		xx[k, :] = numpy.fromstring(fichier_B.readframes(sample_size), numpy.int16) / 65536.0

discr = predict(xx)

print 'average B'
print numpy.average(discr)
