import wave
import numpy
import scipy.spatial.distance as spatial


def energyDst(xs, ys):
	xs = numpy.swapaxes(xs, 0, 1)
	ys = numpy.swapaxes(ys, 0, 1)
	dst = 2*numpy.mean(spatial.cdist(xs, ys, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(xs, xs, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(ys, ys, 'euclidean'))
	return dst
	
def energyDstTheano(xs, ys):
	dim = T.shape(xs)[0]
	dst = 2 * T.sqrt(T.sum(T.sqr(T.reshape(xs, [dim, 1, -1]) - T.reshape(ys, [dim, -1, 1])), [0])).mean([0, 1])
	dst -= T.sqrt(T.sum(T.sqr(T.reshape(xs, [dim, 1, -1]) - T.reshape(xs, [dim, -1, 1])), [0])).mean([0, 1])
	dst -= T.sqrt(T.sum(T.sqr(T.reshape(ys, [dim, 1, -1]) - T.reshape(ys, [dim, -1, 1])), [0])).mean([0, 1])
	return dst


class AudioFileSampler:
        def __init__(self, path, sample_size):
                self.audio_file = wave.open(path)
                self.sample_size = sample_size
                self.max_sample = self.audio_file.getnframes() - sample_size
                
        def batch(self, batch_size):
	        xx = numpy.zeros([1, self.sample_size, batch_size])
	        for k in range(batch_size):
		        self.audio_file.setpos(numpy.random.randint(self.max_sample))
		        x = numpy.fromstring(self.audio_file.readframes(self.sample_size), numpy.int16)
                        x = x.reshape([self.sample_size, 2]) * 1.0 / 65536.0
                        xx[0, :, k] = x[:, 0] + x[:, 1]
                return xx

class AudioFileFreqSampler:
	def __init__(self, path, sample_size, window_size, num):
		self.audio_file = wave.open(path)
		self.sample_size = sample_size
		self.window_size = window_size
                self.max_sample = self.audio_file.getnframes() - sample_size * window_size
		self.num = num
		
	def batch(self, batch_size):
		ff = numpy.zeros([self.num, self.sample_size, batch_size], dtype=numpy.complex128)
		for k in range(batch_size):
		        self.audio_file.setpos(numpy.random.randint(self.max_sample))
		        x = numpy.fromstring(self.audio_file.readframes(self.sample_size * self.window_size), numpy.int16)
                        x = x.reshape([self.sample_size, self.window_size, 2]) * 1.0 / 65536.0
			x = numpy.mean(x, axis=2)
			f = numpy.fft.rfft(x, axis=1)
                        ff[:, :, k] = numpy.swapaxes(f[:, 0:self.num], 0, 1)
		return ff	

	def inverse_transform(self, ff):
		f = numpy.zeros([self.window_size/2 + 1, self.sample_size, ff.shape[2]], dtype=numpy.complex128)
		f[0:self.num, :, :] = ff
		xx = numpy.fft.irfft(f, axis=0)
		xx = numpy.swapaxes(xx, 0, 1).reshape([self.sample_size * self.window_size, -1])
		return xx

	def test_transform(self, batch_size):
		x = numpy.zeros([self.sample_size, self.window_size, batch_size])
		for k in range(batch_size):
		        self.audio_file.setpos(numpy.random.randint(self.max_sample))
		        x0 = numpy.fromstring(self.audio_file.readframes(self.sample_size * self.window_size), numpy.int16)
                        x0 = x0.reshape([self.sample_size, self.window_size, 2]) * 1.0 / 65536.0
			x[:, :, k] = numpy.mean(x0, axis=2)
		
		f = numpy.fft.rfft(x, axis=1)
		num_freq = f.shape[1]

		f[:, num_freq-self.num:num_freq, :] = 0
		xx = numpy.fft.irfft(f, axis=1)

		return (x, f, xx)

		

class SinusSampler:
	def __init__(self, sample_size, noise = 0):
		self.sample_size = sample_size
		self.noise = noise


	def batch(self, batch_size):
		sample_size = self.sample_size
		phase = numpy.random.rand(1, 1, batch_size) * 2 * numpy.pi
		freq = numpy.random.rand(1, 1, batch_size) * 10 + 1
		mag = numpy.random.rand(1, 1, batch_size)
		
		t = numpy.reshape(numpy.linspace(0, 2 * numpy.pi, sample_size), [1, sample_size, 1])
		if self.noise == 0:
			return numpy.sin(freq * t + phase) * mag
		else:
			return numpy.sin(freq * t + phase) * mag + numpy.random.normal(0, self.noise, [1, sample_size, batch_size])



def nearest(x, xx):
	x = numpy.swapaxes(numpy.reshape(x, [-1, x.shape[2]]), 0, 1)
	xx = numpy.swapaxes(numpy.reshape(xx, [-1, xx.shape[2]]), 0, 1)

	nearest = numpy.argmin(spatial.cdist(xx, x, 'euclidean'), 0)
	return nearest

def test_energy_dist(sampler, num_batches=10, num_tests=10):
	result = numpy.zeros([num_tests, num_batches])
	for i in range(num_batches):
		batch_size = (i + 1) * 50
		for j in range(num_tests):
			xx = numpy.reshape(sampler.batch(batch_size), [-1, batch_size])
			xxx = numpy.reshape(sampler.batch(batch_size), [-1, batch_size])
			
			result[j, i] = energyDst(xx, xxx)
	
	return result
		
