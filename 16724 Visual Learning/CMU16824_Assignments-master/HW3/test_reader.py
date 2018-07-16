import pytest
from numpy.testing import assert_allclose
import numpy as np
from reader import textDataLoader, onehot2string

def test_Reader():
	dataloader = textDataLoader('dataset_small.txt')
	character = 'a'

	onehot_vec = np.zeros([128, ], dtype='float32')
	# print(onehot_vec.shape)
	onehot_vec[97] = 1
	# print()
	assert_allclose(dataloader.oneHot(ord(character)), onehot_vec)

	input_b, output_b = dataloader.getBatch(50, 50)
	print(onehot2string(input_b[1]))
	print(onehot2string(output_b[1]))



if __name__ == '__main__':
    pytest.main([__file__])
   