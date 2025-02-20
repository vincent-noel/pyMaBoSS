"""Test suite for uppmaboss simulations."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load, UpdatePopulation
from os.path import dirname, join


class TestUpPMaBoSScMaBoSS(TestCase):

	def test_uppmaboss_cmaboss(self):

		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"))
		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT_cmaboss', cmaboss=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()
		expected_pop_ratios = [
			1.0, 0.6817954975654412, 0.6306956613676165, 0.6114502036053414, 0.5974385752516413, 
			0.5754904027149489, 0.5431751367072919, 0.5029923192031418, 0.46040464153785243, 
			0.4199042244647881, 0.38405999457910445, 0.3541393222851166, 0.3292863788241307
		]

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])

	def test_uppmaboss_cmaboss_only_final_state(self):

		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"))
		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT_cmaboss_final', cmaboss=True, only_final_state=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()
		expected_pop_ratios = [
			1.0, 0.677860000000014, 0.6279830612000052, 0.6097213137802837, 0.5943624338860777, 
			0.5680084035675105, 0.5307981730497715, 0.48714533129812415, 0.4427225485370006, 
			0.40227984372809306, 0.36770389115960767, 0.3393539211511613, 0.31562968852344475
		]

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])

	def test_uppmaboss_cmaboss_for_maboss_load(self):

		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"), cmaboss=True)
		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT_cmaboss', cmaboss=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()
		expected_pop_ratios = [
			1.0, 0.6817954975654412, 0.6306956613676165, 0.6114502036053414, 0.5974385752516413, 
			0.5754904027149489, 0.5431751367072919, 0.5029923192031418, 0.4604046415378524, 
			0.4199042244647881, 0.3840599945791044, 0.3541393222851166, 0.3292863788241307
		]

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])

