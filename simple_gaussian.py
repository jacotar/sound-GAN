import numpy
import theano.tensor as T
import theano

K = 20
KK = 20
N = 30
NN = 5
maxs = 2

batch_size = 50

rng = numpy.random
gen = theano.shared(rng.rand(KK, N) - 0.5)
b_gen = theano.shared(rng.rand(N) - 0.5)
bb_gen = b_gen.dimshuffle('x', 0)

dis1 = theano.shared(rng.rand(N, NN * maxs) - 0.5)
b_dis1 = theano.shared(rng.rand(NN * maxs) - 0.5)
dis2 = theano.shared(rng.rand(NN, 1) - 0.5)
b_dis2 = theano.shared(rng.rand(1) - 0.5)
# bb_dis = b_dis.dimshuffle('x', 0)

x_in = T.dmatrix('x_in')
# batch_size = x_in.shape[0]
z = T.dmatrix('z')

x_gen = T.dot(z, gen) + bb_gen

y_in = T.reshape(T.dot(x_in, dis1) + b_dis1, [batch_size, NN, maxs]).max(2)
y_gen = T.reshape(T.dot(x_gen, dis1) + b_dis1, [batch_size, NN, maxs]).max(2)
discr_in = T.nnet.sigmoid(T.dot(y_in, dis2) + b_dis2)
discr_gen = T.nnet.sigmoid(T.dot(y_gen, dis2) + b_dis2)

cost_in = numpy.log(discr_in).mean()
cost_gen = numpy.log(1 - discr_gen).mean()
cost_gen_wrong = numpy.log(discr_gen).mean()

ggrad_gen = T.grad(-cost_gen, gen)
ggrad_b_gen = T.grad(-cost_gen, b_gen)
ggrad_dis1 = T.grad(cost_in + cost_gen, dis1)
ggrad_b_dis1 = T.grad(cost_in + cost_gen, b_dis1)
ggrad_dis2 = T.grad(cost_in + cost_gen, dis2)
ggrad_b_dis2 = T.grad(cost_in + cost_gen, b_dis2)

lr = 0.05

train_dis = theano.function([x_in, z], 
	[ggrad_gen, ggrad_b_gen, 
	 ggrad_dis1, ggrad_b_dis1, 
	 ggrad_dis2, ggrad_b_dis2, 
	 discr_in, discr_gen, cost_in, cost_gen])
''', 
	updates=[ 
		(dis1, dis1 + lr * grad_dis1), 
		(b_dis1, b_dis1 + lr * grad_b_dis1),
		(dis2, dis2 + lr * grad_dis2), 
		(b_dis2, b_dis2 + lr * grad_b_dis2),
		(gen, gen + 1.5 * lr * grad_gen), 
		(b_gen, b_gen + 1.5 * lr * grad_b_gen)])
'''
train_gen = theano.function([z], [discr_gen, cost_gen], 
	updates=[
		(gen, gen + lr * ggrad_gen), 
		(b_gen, b_gen + lr * ggrad_b_gen)
	])
#generate = theano.function([z], [x_gen])
#discriminate = theano.function([x_in], [discr_in])

A = rng.rand(K, N) - 0.5
b = rng.rand(N) - 0.5
'''
for g in range(200):
	print 'DISCRIMINATOR'
	for t in range(50):
		zz = rng.normal(0, 1.0, [batch_size, K])
		x_in = numpy.dot(zz, A) + b

		z = rng.normal(0, 1.0, [batch_size, KK])
		d_in, d_gen, cost_in, cost_gen = train_dis(x_in, z)
		if t % 50 == 0:
			print (d_in.mean(), d_gen.mean())

	print 'GENERATOR'
	for t in range(50):
		z = rng.normal(0, 1.0, [batch_size, KK])
		d_gen, cost_gen = train_gen(z)

		if t % 10 == 0:
			print d_gen.mean()
'''
max_t = 3000

bs = numpy.zeros([N, max_t]);

num_last = 16;
last_grad_gen = numpy.zeros([num_last, KK*N])
last_grad_b_gen = numpy.zeros([num_last, N])
last_grad_dis1 = numpy.zeros([num_last, N * NN * maxs])
last_grad_b_dis1 = numpy.zeros([num_last, NN * maxs])
last_grad_dis2 = numpy.zeros([num_last, NN])
last_grad_b_dis2 = numpy.zeros([num_last, 1])

last_pos_gen = numpy.zeros([num_last, KK*N])
last_pos_b_gen = numpy.zeros([num_last, N])
last_pos_dis1 = numpy.zeros([num_last, N * NN * maxs])
last_pos_b_dis1 = numpy.zeros([num_last, NN * maxs])
last_pos_dis2 = numpy.zeros([num_last, NN])
last_pos_b_dis2 = numpy.zeros([num_last, 1])


for t in range(max_t):
		zz = rng.normal(0, 1.0, [batch_size, K])
		x_in = numpy.dot(zz, A) + b

		z = rng.normal(0, 1.0, [batch_size, KK])
		
		grad_gen, grad_b_gen, grad_dis1, grad_b_dis1, grad_dis2, grad_b_dis2, d_in, d_gen, cost_in, cost_gen = train_dis(x_in, z)

		norm = numpy.square(grad_gen).sum() + numpy.square(grad_b_gen).sum()
		norm += numpy.square(grad_dis1).sum() + numpy.square(grad_b_dis1).sum()
		norm += numpy.square(grad_dis2).sum() + numpy.square(grad_b_dis2).sum()
		norm = numpy.max(numpy.sqrt(norm), 0.1)

		last_grad_gen[t % num_last, :] = grad_gen.reshape(-1)
		last_grad_b_gen[t % num_last, :] = grad_b_gen
		last_grad_dis1[t % num_last, :] = grad_dis1.reshape(-1)
		last_grad_b_dis1[t % num_last, :] = grad_b_dis1
		last_grad_dis2[t % num_last, :] = grad_dis2.reshape(-1)
		last_grad_b_dis2[t % num_last, :] = grad_b_dis2

		last_pos_gen[t % num_last, :] = gen.get_value().reshape(-1)
		last_pos_b_gen[t % num_last, :] = b_gen.get_value()
		last_pos_dis1[t % num_last, :] = dis1.get_value().reshape(-1)
		last_pos_b_dis1[t % num_last, :] = b_dis1.get_value()
		last_pos_dis2[t % num_last, :] = dis2.get_value().reshape(-1)
		last_pos_b_dis2[t % num_last, :] = b_dis2.get_value()
		

		mean_grad_gen = last_grad_gen.mean(keepdims=True)
		mean_grad_b_gen = last_grad_b_gen.mean(keepdims=True)
		mean_grad_dis1 = last_grad_dis1.mean(keepdims=True)
		mean_grad_b_dis1 = last_grad_b_dis1.mean(keepdims=True)
		mean_grad_dis2 = last_grad_dis2.mean(keepdims=True)
		mean_grad_b_dis2 = last_grad_b_dis2.mean(keepdims=True)

		mean_pos_gen = last_pos_gen.mean(keepdims=True)
		mean_pos_b_gen = last_pos_b_gen.mean(keepdims=True)
		mean_pos_dis1 = last_pos_dis1.mean(keepdims=True)
		mean_pos_b_dis1 = last_pos_b_dis1.mean(keepdims=True)
		mean_pos_dis2 = last_pos_dis2.mean(keepdims=True)
		mean_pos_b_dis2 = last_pos_b_dis2.mean(keepdims=True)

		affinities = (last_pos_gen - mean_pos_gen).dot(grad_gen.reshape(-1)) / norm
		affinities += (last_pos_b_gen - mean_pos_b_gen).dot(grad_b_gen) / norm
		affinities += (last_pos_dis1 - mean_pos_dis1).dot(grad_dis1.reshape(-1)) / norm
		affinities += (last_pos_b_dis1 - mean_pos_b_dis1).dot(grad_b_dis1) / norm
		affinities += (last_pos_dis2 - mean_pos_dis2).dot(grad_dis2.reshape(-1)) / norm
		affinities += (last_pos_b_dis2 - mean_pos_b_dis2).dot(grad_b_dis2) / norm
		
		rot = 0.1
		grad_gen = grad_gen + rot * affinities.T.dot(last_grad_gen - mean_grad_gen).reshape(KK, N)
		grad_b_gen = grad_b_gen + rot * affinities.T.dot(last_grad_b_gen - mean_grad_b_gen)
		grad_dis1 = grad_dis1 + rot * affinities.T.dot(last_grad_dis1 - mean_grad_dis1).reshape(N, NN * maxs)
		grad_b_dis1 = grad_b_dis1 + rot * affinities.T.dot(last_grad_b_dis1 - mean_grad_b_dis1)
		grad_dis2 = grad_dis2 + rot * affinities.T.dot(last_grad_dis2 - mean_grad_dis2).reshape(NN, 1)
		grad_b_dis2 = grad_b_dis2 + rot * affinities.T.dot(last_grad_b_dis2 - mean_grad_b_dis2)
		
		lrr = lr # / numpy.sqrt(t + 2)
		ret = 1.0
		gen.set_value(gen.get_value()*ret + lrr * grad_gen)
		b_gen.set_value(b_gen.get_value()*ret + lrr * grad_b_gen)
		dis1.set_value(dis1.get_value()*ret + lrr * grad_dis1)
		b_dis1.set_value(b_dis1.get_value()*ret + lrr * grad_b_dis1)
		dis2.set_value(dis2.get_value()*ret + lrr * grad_dis2)
		b_dis2.set_value(b_dis2.get_value()*ret + lrr * grad_b_dis2)

		bs[:, t] = b_gen.get_value()
		if t % 50 == 0:
			print (d_in.mean(), d_gen.mean(), numpy.linalg.norm(b_gen.get_value() - b))

max_t = 100
for t in range(max_t):
		zz = rng.normal(0, 1.0, [batch_size, K])
		x_in = numpy.dot(zz, A) + b

		z = rng.normal(0, 1.0, [batch_size, KK])
		
		grad_gen, grad_b_gen, grad_dis1, grad_b_dis1, grad_dis2, grad_b_dis2, d_in, d_gen, cost_in, cost_gen = train_dis(x_in, z)
		
		dis1.set_value(dis1.get_value()*ret + lrr * grad_dis1)
		b_dis1.set_value(b_dis1.get_value()*ret + lrr * grad_b_dis1)
		dis2.set_value(dis2.get_value()*ret + lrr * grad_dis2)
		b_dis2.set_value(b_dis2.get_value()*ret + lrr * grad_b_dis2)

		print (d_in.mean(), d_gen.mean())


import matplotlib.pyplot as P

avg_b = bs.mean(1, keepdims=True)
u, s, v = numpy.linalg.svd(bs - avg_b)

 #P.plot(u[:, 0].dot(bs), u[:, 1].dot(bs))
P.plot(u[:, 0].dot(bs))
P.plot(u[:, 1].dot(bs))
P.plot(u[:, 2].dot(bs))

P.show()
