import numpy
import theano.tensor as T
import theano


num_conv = 5
dimensions = [1, 7, 25, 50, 100, 150]
window_sizes = [8, 8, 4, 4, 4]
max_mults = [3, 3, 3, 3, 3]
'''
num_conv = 1
dimensions = [1, 124]
window_sizes = [2024]
max_mults = [2]

num_conv = 2
dimensions = [1, 25, 124]
window_sizes = [64, 64]
max_mults = [3, 3]
'''

rng = numpy.random
filters = [theano.shared((
		rng.rand(dimensions[i] * window_sizes[i], 
				max_mults[i] * dimensions[i+1])-0.5)/(dimensions[i] * window_sizes[i] * 0.1))
	for i in range(num_conv)]

step_filters = [theano.shared(
		numpy.zeros([dimensions[i] * window_sizes[i], 
				max_mults[i] * dimensions[i+1]], numpy.float64))
	for i in range(num_conv)]

biases = [theano.shared((
	rng.rand(max_mults[i] * dimensions[i+1])-0.5))
	for i in range(num_conv)]

step_biases = [theano.shared(
	numpy.zeros(max_mults[i] * dimensions[i+1], numpy.float64))
	for i in range(num_conv)]

rs = rng.RandomState(1234)
mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))

discrimins = theano.shared((rng.rand(dimensions[num_conv], 1)-0.5)*2.0)
step_discrimins = theano.shared(numpy.zeros([dimensions[num_conv], 1], numpy.float64))

bias_discrimins = theano.shared(0.0)
step_bias_discrimins = theano.shared(0.0)

x = T.dmatrix('x')
sources = T.dcol('sources')
batch_size, num = T.shape(x)

y = x.dimshuffle(0, 1, 'x'); # batch_size, k, dim=1
y = T.unbroadcast(y, 2);

'''
filt = filters[0]*0.0 + 1.0
num = num / window_sizes[0]
source_test = y + sources.dimshuffle(0, 1, 'x')*1.0 - y;
source_test = T.reshape(source_test[:, 0:num*window_sizes[0], :], 
			[batch_size*num, window_sizes[0]*dimensions[0]])
source_test = T.dot(source_test, filt)
source_test = T.reshape(source_test, [batch_size, num, dimensions[1], max_mults[0]])
source_test = source_test.max(3, True)

all_true = source_test[:, 0:10, 1]
#numpy.abs(source_test - 1.0*sources.dimshuffle(0, 1, 'x')).sum().sum().sum()

testing = function([x, sources], [all_true])
'''


for i in range(num_conv):
	num = num / window_sizes[i]
	y = T.reshape(y[:, 0:num*window_sizes[i], :], 
			[batch_size * num, window_sizes[i]*dimensions[i]])
	
	y = T.dot(y, filters[i]) + biases[i]
	
	y = T.switch(mask_rng.binomial(size=y.shape, p=0.8), y, 0)

	y = T.reshape(y, [batch_size, num, dimensions[i+1], max_mults[i]])
	
	y = y.max(3)
	

discr = T.dot(
	T.reshape(y, [batch_size*num, dimensions[num_conv]]),
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

	[(filters[i], filters[i] + 0.01 * step_filters[i]) for i in range(num_conv)] + 
	[(biases[i], biases[i] + 0.01 * step_biases[i]) for i in range(num_conv)] + 
	[(discrimins, discrimins + 0.01 *step_discrimins)] + 
	[(bias_discrimins, bias_discrimins + 0.01 * step_bias_discrimins)]
	)
predict = theano.function([x], [discr])

# training on audio files
import wave

batch_size = 50
sample_size = numpy.product(window_sizes[0:num_conv])

fichier_A = wave.open('Cymbals/Rev_Crsh.wav')
fichier_B = wave.open('Cymbals/Ride_01.wav')

max_A = fichier_A.getnframes() - sample_size
max_B = fichier_B.getnframes() - sample_size

source = numpy.zeros([batch_size*2, 1])
source[batch_size:batch_size*2] = 1
xx = numpy.zeros([batch_size*2, sample_size])

#all_true = testing(xx, source)
#print all_true


for i in range(3000):

	for k in range(batch_size):
		fichier_A.setpos(numpy.random.randint(max_A))
		fichier_B.setpos(numpy.random.randint(max_B))
		xx[k, :] = numpy.fromstring(fichier_A.readframes(sample_size), numpy.int16) / 65536.0
		xx[batch_size + k, :] = numpy.fromstring(fichier_B.readframes(sample_size), numpy.int16) / 65536.0

	discr, cost = train(xx, source)
	print (numpy.max(discr), numpy.min(discr))
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

