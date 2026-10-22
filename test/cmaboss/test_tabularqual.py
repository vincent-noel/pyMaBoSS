"""Test suite for loading TabularQual files with cMaBoSS (mirrors test/test_tabularqual.py)."""

from unittest import TestCase
from maboss import loadTabularQual
from os.path import dirname, join

path = dirname(dirname(__file__))


class TestTabularQualcMaBoSS(TestCase):

	def test_tabularqual(self):

		sim = loadTabularQual(join(path, "Faure2006.xlsx"), cmaboss=True)

		self.assertEqual(
			sorted(list(sim.network.keys())),
			['Cdc20', 'Cdh1', 'CycA', 'CycB', 'CycD', 'CycE', 'E2F', 'Rb', 'UbcH10', 'p27']
		)
