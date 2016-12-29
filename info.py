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
	train_dist = pick.load()


P.figure(1)

P.subplot(221)
P.plot(train_avg[:, 0])
P.plot(train_avg[:, 1])

P.subplot(222)
P.plot(numpy.log(-train_cost[:, 0]))
P.plot(numpy.log(-train_cost[:, 1]))

P.subplot(212)
P.plot(numpy.log(train_dist))

P.show()
