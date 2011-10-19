#!/usr/bin/env python

import sys
from PySide.QtCore import *
from PySide.QtGui import *
import re
from neural_network import NeuralNetwork
from opti_chara_reco import read_trains, read_tests, test_input, TRAINED_FPATH, DESIRED_ACCURACY
import itertools
import random
import thread

GRID_W, GRID_H = 8, 8
GRID_ITEM_W, GRID_ITEM_H = 50, 50
GRID_ITEM_COLORS = [QColor(255, 255, 255), QColor(0, 0, 0)]

SPACING = 10
PANEL_ITEM_W = 100
PANEL_ITEM_H = 50

class Grid(QWidget):
	def __init__(self, parent=None):
		super(Grid, self).__init__(parent)

		self.w = GRID_W
		self.h = GRID_H
		self.clear()

		self.pen_color = 1
		self.prev_x = -1
		self.prev_y = -1

		self.resize(self.w*GRID_ITEM_W, self.h*GRID_ITEM_H)
		self.show()

	def paintEvent(self, ev):
		pa = QPainter(self)

		for i, col in enumerate(self.items):
			for j, item in enumerate(col):
				pa.fillRect(i*GRID_ITEM_W, j*GRID_ITEM_H, GRID_ITEM_W, GRID_ITEM_H, QBrush(GRID_ITEM_COLORS[item]))

	def load_vec(self, vec):
		if len(vec) != self.w*self.h: raise ValueError, 'vector size mismatch'

		self.items = [vec[i:i+self.w*self.h:self.w] for i in xrange(self.w)]
		self.update()

	def clear(self):
		self.items = [[0]*self.h for x in xrange(self.w)]
		self.update()

	def get_vec(self):
		return list(itertools.chain.from_iterable(zip(*self.items)))

	def mousePressEvent(self, ev):
		x = ev.pos().x()/GRID_ITEM_W
		y = ev.pos().y()/GRID_ITEM_H

		try: self.items[x][y] = self.pen_color = int(not self.items[x][y])
		except IndexError: pass

		self.prev_x = x
		self.prev_y = y

		self.update()
		self.parentWidget().test_grid()

	def mouseMoveEvent(self, ev):
		x = ev.pos().x()/GRID_ITEM_W
		y = ev.pos().y()/GRID_ITEM_H

		if self.prev_x == x and self.prev_y == y: return

		try: self.items[x][y] = self.pen_color
		except IndexError: pass

		self.prev_x = x
		self.prev_y = y

		self.update()
		self.parentWidget().test_grid()

class MainWnd(QWidget):
	def __init__(self, parent=None):
		super(MainWnd, self).__init__(parent)

		self.grid = Grid(self)
		self.grid.move(10, 10)

		self.next_g = QPushButton('&Next', self)
		self.next_g.clicked.connect(self.on_next)
		self.next_g.move(self.grid.width() + SPACING*2, SPACING)
		self.next_g.resize(PANEL_ITEM_W, PANEL_ITEM_H)
		self.next_g.show()

		self.prev_g = QPushButton('&Previous', self)
		self.prev_g.clicked.connect(self.on_prev)
		self.prev_g.move(self.grid.width() + SPACING*2, SPACING*2 + PANEL_ITEM_H)
		self.prev_g.resize(PANEL_ITEM_W, PANEL_ITEM_H)
		self.prev_g.show()

		self.clear_g = QPushButton('&Clear', self)
		self.clear_g.clicked.connect(self.on_clear)
		self.clear_g.move(self.grid.width() + SPACING*2, SPACING*3 + PANEL_ITEM_H*2)
		self.clear_g.resize(PANEL_ITEM_W, PANEL_ITEM_H)
		self.clear_g.show()

		self.train_g = QPushButton('&Train', self)
		self.train_g.clicked.connect(self.on_train)
		self.train_g.move(self.grid.width() + SPACING*2, SPACING*4 + PANEL_ITEM_H*3)
		self.train_g.resize(PANEL_ITEM_W, PANEL_ITEM_H)
		self.train_g.show()

		self.out_g = QLabel(self)
		self.out_g.setAlignment(Qt.AlignCenter)
		self.out_g.setFont(QFont(None, 30))
		self.out_g.move(self.grid.width() + SPACING*2, SPACING*5 + PANEL_ITEM_H*4)
		self.out_g.resize(PANEL_ITEM_W, PANEL_ITEM_H)
		self.out_g.show()

		try: self.trains, self.chs = read_trains()
		except IOError: self.trains, self.chs = [], []

		self.n_net = NeuralNetwork()
		try: self.n_net.load_trained_file(TRAINED_FPATH)
		except IOError: pass

		try:
			self.tests = read_tests()
			random.shuffle(self.tests)
		except IOError: self.tests = []
		self.test_idx = -1
		self.cur_test = None
		if self.tests: self.set_test_idx(0)

		self.train_done.connect(self.on_train_done)

		self.setWindowTitle('Optical Character Recognition')
		self.resize(self.grid.width() + SPACING*3 + PANEL_ITEM_W, self.grid.height() + SPACING*2)
		self.show()

	def set_test_idx(self, idx):
		if idx < 0 or idx >= len(self.tests): raise IndexError, 'out of range'

		self.cur_test = self.tests[idx]
		self.test_idx = idx

		self.grid.load_vec(self.cur_test[1])
		self.test_grid()

	def on_next(self):
		try: self.set_test_idx(self.test_idx+1)
		except IndexError: QMessageBox.warning(self, None, 'No more tests.')

	def on_prev(self):
		try: self.set_test_idx(self.test_idx-1)
		except IndexError: QMessageBox.warning(self, None, 'No more tests.')

	def on_clear(self):
		self.grid.clear()
		self.test_grid()

	def on_train(self):
		QMessageBox.warning(self, None, 'Training sequence started. May take several minitues to finish.')

		def proc():
			self.n_net.set_trains(self.trains)
			self.n_net.train(DESIRED_ACCURACY)
			self.n_net.save_trained_file(TRAINED_FPATH)

			self.train_done.emit()

		thread.start_new_thread(proc, ())

	@Slot()
	def on_train_done(self):
		QMessageBox.warning(self, None, 'Training done.')

	train_done = Signal()

	def test_grid(self):
		ch = test_input(self.n_net, self.grid.get_vec(), self.chs)
		self.out_g.setText(ch)

def main():
	app = QApplication(sys.argv)

	wnd = MainWnd()

	app.exec_()

if __name__ == '__main__':
	main()
