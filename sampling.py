import wave
import numpy
import scipy.spatial.distance as spatial


def energyDst(xs, ys):
	dst = 2*numpy.mean(spatial.cdist(xs, ys, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(xs, xs, 'euclidean'))
	dst -= numpy.mean(spatial.cdist(ys, ys, 'euclidean'))
	return dst
	'''
def energyDstTheano(xs, ys):
	dim = T.shape(xs)[2]
	dst = 2 * T.sqrt(T.sum(T.sqr(T.reshape(xs, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])), [2])).mean([0, 1])
	dst -= T.sqrt(T.sum(T.sqr(T.reshape(xs, [1, -1, dim]) - T.reshape(xs, [-1, 1, dim])), [2])).mean([0, 1])
	dst -= T.sqrt(T.sum(T.sqr(T.reshape(ys, [1, -1, dim]) - T.reshape(ys, [-1, 1, dim])), [2])).mean([0, 1])
	return dst

'''
class AudioFileSampler:
        def __init__(self, path, sample_size):
                self.audio_file = wave.open(path)
                self.sample_size = sample_size
                self.max_sample = self.audio_file.getnframes() - sample_size
                
        def batch(self, batch_size):
	        xx = numpy.zeros([batch_size, self.sample_size, 1])
	        for k in range(batch_size):
		        self.audio_file.setpos(numpy.random.randint(self.max_sample))
		        x = numpy.fromstring(self.audio_file.readframes(self.sample_size), numpy.int16)
                        x = numpy.reshape(x, [self.sample_size, 2]) * 1.0 / 65536.0
                        xx[k, :, 0] = x[:, 0] + x[:, 1]
                return xx

class AudioFileFreqSampler:
	def __init__(self, path, sample_size, window_size, num):
		
		self.audio_sampler = AudioFileSampler(path, sample_size * window_size)
		self.sample_size = sample_size
		self.window_size = window_size
		self.num_freq = window_size/2 + 1
		self.num = num
		
	def batch(self, batch_size):
		xx = self.audio_sampler.batch(batch_size)
		f = numpy.fft.rfft(xx.reshape([batch_size, self.sample_size, self.window_size]), axis=2)
	
		ff = numpy.zeros([batch_size, self.sample_size, 2*self.num])
                ff[:, :, 0:self.num] = f[:, :, 1:self.num+1].real
                ff[:, :, self.num:self.num*2] = f[:, :, 1:self.num+1].imag
		return ff, xx

	def inverse_transform(self, ff):
		f = numpy.zeros([ff.shape[0], self.sample_size, self.num_freq], dtype=numpy.complex128)
		f[:, :, 1:self.num+1] = ff[:, :, 0:self.num] + 1.j * ff[:, :, self.num:self.num*2]
		xx = numpy.fft.irfft(f, axis=2)
		xx = xx.reshape([-1, self.sample_size * self.window_size])
		return xx


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



def nearest(x, xx):
	x = numpy.reshape(x, [x.shape[0], -1])
	xx = numpy.reshape(xx, [xx.shape[0], -1])

	nearest = numpy.argmin(spatial.cdist(xx, x, 'euclidean'), 0)
	return nearest

def test_energy_dist(sampler, num_batches=10, num_tests=10):
	result = numpy.zeros([num_batches, num_tests])
	for i in range(num_batches):
		batch_size = (i + 1) * 50
		for j in range(num_tests):
			xx = numpy.reshape(sampler.batch(batch_size), [batch_size, -1])
			xxx = numpy.reshape(sampler.batch(batch_size), [batch_size, -1])
			
			result[j, i] = energyDst(xx, xxx)
	
	return result

