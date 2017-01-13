import wave
import numpy
import scipy.spatial.distance as spatial

import theano.tensor as T

import pickle

def energyDst(xs, ys):
	dst = 2*numpy.mean(spatial.cdist(xs, ys, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(xs, xs, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(ys, ys, 'euclidean'))
	return dst


def energyDstTheano(xs, ys, L=2):
	dim = T.shape(xs)[2]
	dst = 2 * (T.reshape(xs, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])).norm(L, axis=[2]).mean([0, 1])
	dst -= (T.reshape(xs, [1, -1, dim]) - T.reshape(xs, [-1, 1, dim])).norm(L, [2]).mean([0, 1])
	dst -= (T.reshape(ys, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])).norm(L, [2]).mean([0, 1])
	return dst

def energyDstTheano2(xs, ys):
	dim = T.shape(xs)[2]
	dst = 2 * T.sqrt(T.sqr(T.reshape(xs, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])).sum(axis=[2])+0.01).mean([0, 1])
	dst -= T.sqrt(T.sqr(T.reshape(xs, [1, -1, dim]) - T.reshape(xs, [-1, 1, dim])).sum([2])+0.01).mean([0, 1])
	dst -= T.sqrt(T.sqr(T.reshape(ys, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])).sum([2])+0.01).mean([0, 1])
	return dst


class AudioFileSampler:
        def __init__(self, paths, sample_size, window_size=1024, num=32):
                self.num_files = len(paths)
		self.paths = paths
		self.audio_file = [wave.open(p) for p in paths]
                self.sample_size = sample_size
                self.max_sample = [af.getnframes() - sample_size * window_size for af in self.audio_file]

class AudioFileSampler:
        def __init__(self, paths, sample_size, window_size=1024, num=32):
                self.num_files = len(paths)
		self.paths = paths
		self.audio_file = [wave.open(p) for p in paths]
                self.sample_size = sample_size
                self.max_sample = [af.getnframes() - sample_size * window_size for af in self.audio_file]
		self.window_size = window_size
		self.num_freq = window_size/2 + 1
		self.num = num
		
		self.mults = numpy.ones(num*2)

		ff = self.toFreq(self.batch(200))
		self.avg = ff.mean(axis=(0, 1))
		self.var = numpy.square(ff - self.avg).mean(axis=(0, 1))
		self.mults = 1 / (0.1 + numpy.sqrt(self.var))
	
                
        def batch(self, batch_size):
	        xx = numpy.zeros([batch_size, self.sample_size, self.window_size, 1])
	        for k in range(batch_size):
			i = numpy.random.randint(self.num_files)
			
		        self.audio_file[i].setpos(numpy.random.randint(self.max_sample[i]))
		        x = numpy.fromstring(self.audio_file[i].readframes(
				self.sample_size*self.window_size), numpy.int16)
                        x = numpy.reshape(x, [self.sample_size, self.window_size, 2]) * 1.0 / 65536.0
                        xx[k, :, :] = numpy.mean(x, 2, keepdims=True)
                return xx

	def toFreq(self, xx):
		batch_size = xx.shape[0]
		f = numpy.fft.rfft(xx, axis=2)
		f = numpy.concatenate(
			[f[:, :, 0:self.num].real, 
			f[:, :, 0:self.num].imag], 2)
	
		return numpy.reshape(f, [batch_size, self.sample_size, self.num*2]) * self.mults
		
	def fromFreq(self, ff):
		ff = ff / self.mults
		f = numpy.zeros([ff.shape[0], self.sample_size, self.num_freq], dtype=numpy.complex128)
		f[:, :, 0:self.num] = ff[:, :, 0:self.num] + 1.j * ff[:, :, self.num:self.num*2]
		xx = numpy.fft.irfft(f, axis=2)
		#xx = xx.reshape([-1, self.sample_size * self.window_size])
		return xx
	def save(self, path):
		with open(path, "wb") as f:
			pick = pickle.Pickler(f)
			
			pick.dump(self.paths)
			pick.dump(self.sample_size)
			pick.dump(self.window_size)
			pick.dump(self.num)
	@classmethod
	def load(self, path):
		with open(path, "rb") as f:
			pick = pickle.Unpickler(f)
			self = self(
				pick.load(),
				pick.load(),
				pick.load(),
				pick.load()
				)
		return self

class SinusSampler:
	def __init__(self, sample_size, noise = 0):
		self.sample_size = sample_size
		self.noise = noise


	def batch(self, batch_size):
		sample_size = self.sample_size
		phase = numpy.random.rand(batch_size, 1, 1) * 2 * numpy.pi
		freq = numpy.random.rand(batch_size, 1, 1) * 10 + 1
		mag = numpy.random.rand(batch_size, 1, 1)
		
		t = numpy.reshape(numpy.linspace(0, 2 * numpy.pi, sample_size), [1, sample_size, 1])
		if self.noise == 0:
			return numpy.sin(freq * t + phase) * mag
		else:
			return numpy.sin(freq * t + phase) * mag + numpy.random.normal(0, self.noise, [batch_size, sample_size, 1])

class GaussianProcess:
	def __init__(self, sample_size, filt, drop=1.0):
		self.sample_size = sample_size
		self.filt = filt
		self.drop = drop


	def batch(self, batch_size):
		win, d = self.filt.shape
		x0 = numpy.random.normal(0, 1, [batch_size, self.sample_size+win-1, d])
		x0 *= numpy.random.binomial(1, self.drop, [batch_size, self.sample_size+win-1, 1])
		xx = numpy.zeros([batch_size, self.sample_size, 1])
		for i in range(batch_size):
			for j in range(d):
				xx[i, :, 0] += numpy.convolve(x0[i, :, j], self.filt[:, j], 'valid')
		
		return xx

	def toFreq(self, xx):
		return xx

	def save(self, path):
		with open(path, "wb") as f:
			pick = pickle.Pickler(f)
			
			pick.dump(self.sample_size)
			pick.dump(self.filt)
			pick.dump(self.drop)

import types

def fromPickle(path):
	with open(path, "rb") as f:
		pick = pickle.Unpickler(f)
		paths_or_sample_size = pick.load()
		
		if type(paths_or_sample_size) == numpy.int64:
			sampler = GaussianProcess(paths_or_sample_size, pick.load(), pick.load())
		else:
			sampler = AudioFileSampler(paths_or_sample_size, pick.load(), pick.load(), pick.load())

		return sampler


def nearest(x, xx):
	x = numpy.reshape(x, [x.shape[0], -1])
	xx = numpy.reshape(xx, [xx.shape[0], -1])

	nearest = numpy.argmin(spatial.cdist(xx, x, 'euclidean'), 0)
	return nearest

def test_energy_dist(sampler, num_batches=10, num_tests=10):
	result = numpy.zeros([num_batches, num_tests])
	for i in range(num_batches):
		batch_size = (i + 0.5) * 20
		for j in range(num_tests):
			xx = numpy.reshape(sampler.toFreq(sampler.batch(batch_size)), [batch_size, -1])
			xxx = numpy.reshape(smapler.toFreq(sampler.batch(batch_size)), [batch_size, -1])
			
			result[j, i] = energyDst(xx, xxx)

		print numpy.mean(result[:, i])
	
	return result

