#!/usr/bin/env python

import random
import math
import re

sigmoid = lambda x: 1/(1+math.exp(-x))
step = lambda x: 1 if x >= 0 else 0

class NeuralNetwork:
	def __init__(self):
		self.learning_rate = 0.01
		self.trains = None
		self.weights = None

	def init_vecs(self, neuron_cnts):
		if not neuron_cnts: raise ValueError, 'empty neuron set'

		self.neuron_cnts = neuron_cnts
		self.weights = [[[0]*(self.neuron_cnts[i]+1) for j in xrange(neuron_cnt)] for i, neuron_cnt in enumerate(self.neuron_cnts[1:])]
		self.inputs = [[0]*neuron_cnt for neuron_cnt in self.neuron_cnts]
		self.deltas = [[0]*neuron_cnt for neuron_cnt in self.neuron_cnts[1:]]

	def reset_weights(self):
		for i, neuron_cnt in enumerate(self.neuron_cnts[1:]):
			for j in xrange(neuron_cnt):
				for k in xrange(self.neuron_cnts[i]+1):
					self.weights[i][j][k] = random.random()

	def set_trains(self, trains):
		self.trains = trains

		self.init_vecs([len(self.trains[0][0]), 3, 3, 3])
		self.reset_weights()

	def load_trained_file(self, fpath):
		fp = open(fpath)
		self.init_vecs([int(x) for x in re.findall('[0-9]+', fp.readline())])

		for i, neuron_cnt in enumerate(self.neuron_cnts[1:]):
			for j in xrange(neuron_cnt):
				for k in xrange(self.neuron_cnts[i]+1):
					self.weights[i][j][k] = float(fp.readline())

	def save_trained_file(self, fpath):
		fp = open(fpath, 'w')
		fp.write('%s\n' % ' '.join(str(x) for x in self.neuron_cnts))

		for i, neuron_cnt in enumerate(self.neuron_cnts[1:]):
			for j in xrange(neuron_cnt):
				for k in xrange(self.neuron_cnts[i]+1):
					fp.write('%.20f\n' % self.weights[i][j][k])

	def fill_inputs(self, vec, func):
		if not self.weights: raise ValueError, 'not yet trained'

		self.inputs[0] = vec

		for i, neuron_cnt in enumerate(self.neuron_cnts[1:]):
			for j in xrange(neuron_cnt):
				self.inputs[i+1][j] = func(sum(self.weights[i][j][k]*(self.inputs[i][k] if k < self.neuron_cnts[i] else 1) for k in xrange(self.neuron_cnts[i]+1)))

	def calc_deltas(self, desired):
		for j in xrange(self.neuron_cnts[-1]):
			self.deltas[-1][j] = (desired[j]-self.inputs[-1][j])*self.inputs[-1][j]*(1-self.inputs[-1][j])

		for i, neuron_cnt in reversed(list(enumerate(self.neuron_cnts[1:-1]))):
			for j in xrange(neuron_cnt):
				self.deltas[i][j] = self.inputs[i+1][j]*(1-self.inputs[i+1][j])*sum(self.deltas[i+1][k]*self.weights[i+1][k][j] for k in xrange(self.neuron_cnts[i+2]))

	def update_weights(self):
		for i, neuron_cnt in enumerate(self.neuron_cnts[1:]):
			for j in xrange(neuron_cnt):
				for k in xrange(self.neuron_cnts[i]+1):
					self.weights[i][j][k] += self.learning_rate*self.deltas[i][j]*(self.inputs[i][k] if k < self.neuron_cnts[i] else 1)

	def calc_err(self):
		err = 0

		for train in self.trains:
			self.fill_inputs(train[1], sigmoid)

			desired = train[0]
			err += sum((desired[i]-self.inputs[-1][i])*(desired[i]-self.inputs[-1][i]) for i in xrange(len(self.inputs[-1])))

		return err/(len(self.trains)*len(self.inputs[-1]))

	def epoch(self):
		if not self.trains: raise ValueError, 'training set not exists'

		for train in self.trains:
			self.fill_inputs(train[1], sigmoid)
			self.calc_deltas(train[0])
			self.update_weights()

		return self.calc_err()

	def train(self, desired_acc):
		self.reset_weights()

		while self.epoch() > desired_acc: pass

	def test_input(self, vec):
		self.fill_inputs(vec, sigmoid)
		return self.inputs[-1]
