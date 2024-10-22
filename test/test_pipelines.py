"""Test suite for simulation probability distributions."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')
import pandas, numpy

from unittest import TestCase
from maboss import load, set_nodes_istate
from maboss.pipelines import simulate_single_mutants, simulate_double_mutants, filter_sensitivity
from os.path import dirname, join


class TestPipelines(TestCase):

	def test_sensitivity_cellcycle(self):

		path = dirname(__file__)
		sim = load(
			join(path, "cellcycle.bnd"), 
			join(path, "cellcycle_runcfg.cfg"), 
			join(path, "cellcycle_runcfg-thread_1.cfg")
		)
		
		ress = simulate_single_mutants(sim, sign="OFF")
		self.assertEqual(
			list(ress.keys()), 
			[('CycD', 'OFF'), ('CycE', 'OFF'), ('CycA', 'OFF'), ('CycB', 'OFF'), ('Rb', 'OFF'), ('E2F', 'OFF'), ('p27', 'OFF'), ('Cdc20', 'OFF'), ('UbcH10', 'OFF'), ('cdh1', 'OFF')]
		)
		
		self.assertTrue(numpy.isclose(
			ress[('CycD', 'OFF')].get_last_states_probtraj().values[0],
			numpy.array([9.99949e-01, 1.60000e-05, 1.50000e-05, 1.80000e-05, 1.00000e-06, 1.00000e-06])
		).all())
		
		self.assertTrue(numpy.isclose(
			ress[('CycD', 'OFF')].get_last_nodes_probtraj().values[0],
			numpy.array([3.2e-05, 3.3e-05, 2.0e-06])
		).all())
		
		fres = filter_sensitivity(ress, "<nil>", maximum=0.5)
		self.assertEqual(
			list(fres.keys()),
			[('CycE', 'OFF'), ('CycA', 'OFF'), ('CycB', 'OFF'), ('Rb', 'OFF'), ('E2F', 'OFF'), ('p27', 'OFF'), ('Cdc20', 'OFF'), ('UbcH10', 'OFF'), ('cdh1', 'OFF')]
		)
		
		resd = simulate_double_mutants(sim, list_nodes=['CycD', 'CycA'])
		self.assertEqual(
			list(resd.keys()), 
			[(('CycD', 'ON'), ('CycA', 'ON')), (('CycD', 'ON'), ('CycA', 'OFF')), (('CycA', 'ON'), ('CycD', 'OFF')), (('CycD', 'OFF'), ('CycA', 'OFF'))]
		)