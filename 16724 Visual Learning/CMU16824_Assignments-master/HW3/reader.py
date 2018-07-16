'''
This file contains the class for reading text files.
Please fill up the functions oneHot and convertHot.
'''
import numpy as np
import random
class textDataLoader(object):
	def __init__(self, datapath):
		self.nFeats = 128
		print 'Loading Data'
		self.D = open(datapath,'r').read()
		self.nChars = len(self.D)
		self.N = [ord(c) for c in self.D]
		# print(self.N)
		# print(max(self.N))
		self.D_oneHot = np.array([self.oneHot(n) for n in self.N])
		# print(self.D_oneHot.shape)
		# time.sleep(1000)

	def oneHot(self, character):
		'''
		In this function you need to output a one hot encoding of the ASCII character.
		'''
		one_hot_vec = np.zeros([self.nFeats, ], dtype='float32')
		# print(character)
		one_hot_vec[character] = 1
		return one_hot_vec
	
	def convertHot(self, string_l):
		'''
		In this function, you will need to write a piece of code that converts a string
		to a numpy array of one hot representation.
		'''
		one_hot_vec = [self.oneHot(ord(string_l[0]))]
		for character in string_l[1:]:
			one_hot_vec = np.vstack((one_hot_vec, [self.oneHot(ord(character))]))
		return one_hot_vec

	def getBatch(self, batch_size, max_length):
		input_b = np.zeros([batch_size, max_length, self.nFeats])
		output_b =  np.zeros([batch_size, max_length, self.nFeats])
		for i in range(batch_size):
			r = random.randint(0, self.nChars-2-max_length)
			input_b[i] = self.D_oneHot[r:r+max_length]
			output_b[i] = self.D_oneHot[r+1:r+1+max_length]
		return input_b, output_b


def onehot2ord(onehot):
	'''
	convert one hot vector to a ord integer number
	'''
	assert isinstance(onehot, np.ndarray) and onehot.ndim == 1, 'input should be 1-d numpy array'
	assert sum(onehot) == 1 and np.count_nonzero(onehot) == 1, 'input numpy array is not one hot vector'
	return np.argmax(onehot)

def onehot2character(onehot):
	'''
	convert one hot vector to a character
	'''
	return chr(onehot2ord(onehot))


def onehot2string(onehot):
	'''
	convert a set of one hot vector to a string
	'''
	if isinstance(onehot, np.ndarray):
		onehot.ndim == 2, 'input should be 2-d numpy array'
		onehot = list(onehot)
	elif isinstance(onehot, list):
		assert CHECK_EQ_LIST([tmp.ndim for tmp in onehot]), 'input list of one hot vector should have same length'
	else:
		assert False, 'unknown error'

	assert all(sum(onehot_tmp) == 1 and np.count_nonzero(onehot_tmp) == 1 for onehot_tmp in onehot), 'input numpy array is not a set of one hot vector'
	ord_list = [onehot2ord(onehot_tmp) for onehot_tmp in onehot]
	return ord2string(ord_list)


def ord2string(ord_list):
	'''
	convert a list of ASCII character to a string
	'''
	assert isinstance(ord_list, list) and len(ord_list) > 0, 'input should be a list of ord with length larger than 0'
	assert all(isinteger(tmp) for tmp in ord_list), 'all elements in the list of ord should be integer'
	
	L = ''
	for o in ord_list:
		L += chr(o)
	
	return L


def isinteger(integer_test):
	return isinstance(integer_test, int)
