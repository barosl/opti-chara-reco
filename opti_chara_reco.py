#!/usr/bin/env python

import re
from neural_network import NeuralNetwork
import os

TRAINING_SET_FPATH = 'input/traindata.txt'
TEST_SET_FPATH = 'input/testdata.txt'
TRAINED_FPATH = 'trained.txt'
DESIRED_ACCURACY = 0.01

def read_trains():
	trains = []
	chs = []
	ch_cache = {}

	for line in open(TRAINING_SET_FPATH).read().splitlines():
		words = re.findall('[^\\s$]+', line)
		ch = words[0]
		vec = [int(x) for x in words[1:-1]]

		if ch not in ch_cache:
			ch_cache[ch] = len(chs)
			chs.append(ch)

		trains.append([ch_cache[ch], vec])

	if not chs: raise ValueError, 'empty training set'

	desired_list = [[0]*i+[1]+[0]*(len(chs)-1-i) for i in xrange(len(chs))]

	for train in trains:
		train[0] = desired_list[train[0]]

	return trains, chs

def read_tests():
	tests = []

	for line in open(TEST_SET_FPATH).read().splitlines():
		words = re.findall('[^\\s$]+', line)
		ch = words[0]
		vec = [int(x) for x in words[1:-1]]

		tests.append([ch, vec])

	return tests


def train(n_net, desired_acc):
	idx = 0
	prev_err = -1

	while True:
		err = n_net.epoch()

		if prev_err != -1 and prev_err < err: print '(reversed)',
		prev_err = err

		idx = idx+1
		print '%d. %.7f' % (idx, err)

		if err <= desired_acc: break

def test_input(n_net, vec, chs):
	return chs[max(enumerate(n_net.test_input(vec)), key=lambda x: x[1])[0]]

def test_inputs(n_net, chs):
	total = 0
	succ = 0

	tests = read_tests()
	if not tests: raise ValueError, 'test not exists'

	for ch, vec in tests:
		total += 1
		if test_input(n_net, vec, chs) == ch: succ += 1

	print 'Accuracy: %.1f%%' % (100.*succ/total)

def main():
	trains, chs = read_trains()

	n_net = NeuralNetwork()

	if os.path.exists(TRAINED_FPATH):
		n_net.load_trained_file(TRAINED_FPATH)
	else:
		n_net.set_trains(trains)
		n_net.train(DESIRED_ACCURACY)
		n_net.save_trained_file(TRAINED_FPATH)

	test_inputs(n_net, chs)

if __name__ == '__main__':
	main()
