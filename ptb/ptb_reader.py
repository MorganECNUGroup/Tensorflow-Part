"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange #pylint:disable=redefined-buildin
import tensorflow as tf

from tensorflow.python.platform import gfile

def _read_words(filename):
	with gfile.GFile(filename,"r") as f:
		return f.read().replace("\n","<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(),key = lambda x:-x[1])

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words,range(len(words))))

	return word_to_id

def _file_to_word_ids(filename,word_to_id):
	data = _read_words(filename)
	return [word_to_id[word] for word in data]

def ptb_raw_data(data_path = None):
	""""""
	train_path = os.path.join(data_path,"ptb.train.txt")
	valid_path = os.path.join(data_path,"ptb.valid.txt")
	test_path = os.path.join(data_path,"ptb.test.txt")

	print("reading training data from",train_path)
	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path,word_to_id)
	valid_data = _file_to_word_ids(valid_path,word_to_id)
	test_data = _file_to_word_ids(test_path,word_to_id)
	vocabulary = len(word_to_id)
	id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
	print("first word is",id_to_word[1])
	return train_data,valid_data,test_data,word_to_id,id_to_word

def ptb_iterator(raw_data,batch_size,num_steps):
	"""Iterata on the raw PTB data"""
	raw_data = np.array(raw_data,dtype = np.int32)
	data_len = len(raw_data)
	batch_len = data_len//batch_size
	data = np.zeros([batch_size,batch_len],dtype = np.int32)
	for i in range(batch_size):
		data[i] = raw_data[batch_len*i:batch_len*(i+1)]
	epoch_size = (batch_len-1)//num_steps

	# print("now we are in the ptb_iterator")
	# print("the epoch_size is:")
	# print(epoch_size)
	# print("the data is")
	# print(data)
	if epoch_size == 0:
		raise ValueError("epoch_size == 0,decrease batch_size or num_steps")
	for i in range(epoch_size):
		x = data[:,i*num_steps:(i+1)*num_steps]
		y = data[:,i*num_steps+1:(i+1)*num_steps+1]
		yield(x,y) 