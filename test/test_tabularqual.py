"""Test suite for loading TabularQual files."""

#For testing in environnement with no screen
import numpy

from unittest import TestCase
from maboss import loadTabularQual
from os.path import dirname, join


class TestTabularQual(TestCase):

	def test_tabularqual(self):

		path = dirname(__file__)
		sim = loadTabularQual(join(path, "Faure2006.xlsx"))
		
		self.assertEqual(
			sorted(list(sim.network.keys())), 
			['Cdc20', 'Cdh1', 'CycA', 'CycB', 'CycD', 'CycE', 'E2F', 'Rb', 'UbcH10', 'p27']
		)	