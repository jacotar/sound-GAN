import numpy

import sys

import matplotlib.pyplot as P

import pickle

path = sys.argv[1]

with open(path+"/info.p", "rb") as f:
	pick = pickle.Unpickler(f)

	train_avg = pick.load()
	train_cost = pick.load()
	train_ranks = pick.load()
	train_percentiles = pick.load()
	train_dist = pick.load()
'''
with open(path+"/info.p", "rb") as f:
	pick = pickle.Unpickler(f)

	train_avg = numpy.concatenate((train_avg, pick.load()))
	train_cost = numpy.concatenate((train_cost, pick.load()))
	train_ranks = pick.load()
	train_percentiles = numpy.concatenate((train_percentiles, pick.load()))
	train_dist = numpy.concatenate((train_dist, pick.load()))
'''


P.figure(1)

#P.subplot(321)
P.plot(train_avg[:, 0])
P.plot(train_avg[:, 1])

P.figure(2)
#P.subplot(322)
P.semilogy(-train_cost[:, 0])
P.semilogy(-train_cost[:, 1])

P.figure(3)
#P.subplot(323)
P.semilogy(train_dist)

P.figure(4)
P.subplot(121)
for p in range(5):
	#P.plot(numpy.log(1 - train_percentiles[:, 0, p]))
	P.plot(train_percentiles[:, 0, p])

P.subplot(122)
for p in range(5):
	P.plot(train_percentiles[:, 1, p])

P.show()
