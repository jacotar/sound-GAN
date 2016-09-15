import numpy
import theano.tensor as T
import theano
import theano.typed_list

x = T.dmatrix('x')
sources = T.dcol('sources')
batch_size, k = T.shape(x)

num_conv = 3
dimensions = [1, 7, 40, 60, 100]
window_sizes = [8, 8, 3, 3]
strides = [8, 8, 3, 3]
max_mults = [2, 2, 3, 3]

rng = numpy.random
filters = [theano.shared(
		rng.rand(max_mults[i], dimensions[i+1], 
				dimensions[i], window_sizes[i])-0.5)
	for i in range(num_conv)
	]
biases = [theano.shared(
	rng.rand(max_mults[i], dimensions[i+1])-0.5)
	for i in range(num_conv)
	]

discrimins = theano.shared(rng.rand(dimensions[num_conv], 1)-0.5)
bias_discrimins = theano.shared(0.0)

y = x.dimshuffle(0, 'x', 'x', 1);
y = T.unbroadcast(y, 1);
y = T.unbroadcast(y, 2);

discr = 0;
cost = 0;

for i in range(num_conv):
	filt = T.reshape(filters[i], 
		[dimensions[i+1] * max_mults[i], dimensions[i],
		1, window_sizes[i]])
	bias = T.reshape(biases[i], [dimensions[i+1] * max_mults[i]]);
	bias = bias.dimshuffle('x', 0, 'x', 'x')
	y = T.nnet.conv.conv2d(y, filt, subsample=(strides[i], 1)) + bias
	

	num = T.shape(y)[3]
	y = T.reshape(y, [batch_size, dimensions[i+1], max_mults[i], num])
	y = y.max(2, True)
	
	#discrs = T.dot(
	#	T.reshape(y.dimshuffle(0, 3, 1), [batch_size*num, dimensions[i+1]]),
	#	discrimins[i])
	#discrs = T.nnet.sigmoid(T.reshape(discrs, [batch_size, num]) + bias_discrimins[i])
	#discr = discr + 0.00001 * discrs.sum(1) * discr_weights[i]
	#cost = cost + 0.00001 * numpy.log(numpy.abs(sources - discrs)).sum().sum() * discr_weights[i]

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
	[(filters[i], filters[i] + 0.002 * grad_filters[i]) for i in range(num_conv)] + 
	[(biases[i], biases[i] + 0.002 * grad_biases[i]) for i in range(num_conv)] + 
	[(discrimins, discrimins + 0.002 *grad_discrimins)] + 
	[(bias_discrimins, bias_discrimins + 0.002 * grad_bias_discrimins)]
	)
predict = theano.function([x], [discr])

# training on audio files
import wave

batch_size = 30
sample_size = 1024

fichier_A = wave.open('Claps/Bg_Clp.wav')
fichier_B = wave.open('Cymbals/Ride_01.wav')

max_A = fichier_A.getnframes() - sample_size
max_B = fichier_B.getnframes() - sample_size

source = numpy.zeros([batch_size*2, 1])
source[batch_size:batch_size*2] = 1;
xx = numpy.zeros([batch_size*2, sample_size])

for i in range(20):
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
