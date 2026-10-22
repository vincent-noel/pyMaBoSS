"""Test suite for pipelines on cMaBoSS simulations (mirrors test/test_pipelines.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')
import pandas, numpy

from unittest import TestCase
from maboss import load
from maboss.pipelines import simulate_single_mutants, simulate_double_mutants, filter_sensitivity
from os.path import dirname, join

path = dirname(dirname(__file__))


class TestPipelinescMaBoSS(TestCase):

	def test_sensitivity_cellcycle(self):

		sim = load(
			join(path, "cellcycle.bnd"),
			join(path, "cellcycle_runcfg.cfg"),
			join(path, "cellcycle_runcfg-thread_1.cfg"),
			cmaboss=True
		)

		ress = simulate_single_mutants(sim, sign="OFF")
		self.assertEqual(
			list(ress.keys()),
			[('CycD', 'OFF'), ('CycE', 'OFF'), ('CycA', 'OFF'), ('CycB', 'OFF'), ('Rb', 'OFF'), ('E2F', 'OFF'), ('p27', 'OFF'), ('Cdc20', 'OFF'), ('UbcH10', 'OFF'), ('cdh1', 'OFF')]
		)

		# Same values as test/test_pipelines.py, compared by label as cMaBoSS orders the columns differently
		last_states = ress[('CycD', 'OFF')].get_last_states_probtraj().iloc[0]
		last_states = last_states[last_states > 0]
		self.assertEqual(len(last_states), 6)
		self.assertTrue(numpy.isclose(
			sorted(last_states.values),
			sorted([9.99949e-01, 1.60000e-05, 1.50000e-05, 1.80000e-05, 1.00000e-06, 1.00000e-06])
		).all())
		self.assertAlmostEqual(last_states["<nil>"], 9.99949e-01)

		last_nodes = ress[('CycD', 'OFF')].get_last_nodes_probtraj().iloc[0]
		last_nodes = last_nodes[last_nodes > 0]
		self.assertTrue(numpy.isclose(
			sorted(last_nodes.values),
			sorted([3.2e-05, 3.3e-05, 2.0e-06])
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
