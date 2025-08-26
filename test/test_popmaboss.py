"""Test suite for popmaboss simulations."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import PopSimulation
from os.path import dirname, join
import numpy as np

class TestPopPMaBoSS(TestCase):

	def test_assymetric(self):

		sim = PopSimulation(join(dirname(__file__), "pop", "Assymetric.pbnd"), join(dirname(__file__), "pop", "Assymetric.cfg"))
		res = sim.run()

		expected = np.array([
			[1.00019671, 0.12471371, 0.00476415, 0.12019963],
			[1.00231314, 0.37357459, 0.03085417, 0.34311544],
			[1.00886292, 0.62094206, 0.0759272,  0.54396029],
			[1.02004758, 0.86600702, 0.13632403, 0.72698386],
			[1.03604263, 1.10977551, 0.20610156, 0.89715887],
			[1.05656014, 1.3525056,  0.28416642, 1.05669773],
			[1.08048681, 1.59472274, 0.36860565, 1.20572626],
			[1.10811786, 1.83552332, 0.45716192, 1.34805735],
			[1.13815353, 2.07554602, 0.54958219, 1.48536074],
			[1.17027793, 2.31753231, 0.64445287, 1.61791143]
		])
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, rtol=1e-1, atol=1e-2))
		
		expected = np.array([6.56597326e-01, 2.56412951e-01, 7.06043899e-02, 1.34984659e-02, 2.61518102e-03, 2.71686839e-04])
		self.assertTrue(np.allclose(res.get_last_state_dist("B -- C").values, expected, rtol=5e-2, atol=1e-2))
		
	def test_assymetric_restore(self):

		sim = PopSimulation(join(dirname(__file__), "pop", "Assymetric.pbnd"), join(dirname(__file__), "pop", "Assymetric.cfg"))
		res = sim.run(workdir="pop", prefix="res_assymetric", overwrite=False)

		expected = np.array([
			[1.00019671, 0.12471371, 0.00476415, 0.12019963],
			[1.00231314, 0.37357459, 0.03085417, 0.34311544],
			[1.00886292, 0.62094206, 0.0759272,  0.54396029],
			[1.02004758, 0.86600702, 0.13632403, 0.72698386],
			[1.03604263, 1.10977551, 0.20610156, 0.89715887],
			[1.05656014, 1.3525056,  0.28416642, 1.05669773],
			[1.08048681, 1.59472274, 0.36860565, 1.20572626],
			[1.10811786, 1.83552332, 0.45716192, 1.34805735],
			[1.13815353, 2.07554602, 0.54958219, 1.48536074],
			[1.17027793, 2.31753231, 0.64445287, 1.61791143]
		])
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, rtol=1e-1, atol=1e-2))
		
		expected = np.array([6.56597326e-01, 2.56412951e-01, 7.06043899e-02, 1.34984659e-02, 2.61518102e-03, 2.71686839e-04])
		self.assertTrue(np.allclose(res.get_last_state_dist("B -- C").values, expected, rtol=5e-2, atol=1e-2))
		
	def test_fork(self):

		sim = PopSimulation(join(dirname(__file__), "pop", "Fork.bnd"), join(dirname(__file__), "pop", "Fork.pcfg"))
		res = sim.run()
		
		expected = np.array([
			[1.29723970e+00, 8.51428818e-01, 8.51331484e-01],
			[1.75635421e-01, 1.41222465e+00, 1.41213993e+00],
			[2.35995150e-02, 1.48814640e+00, 1.48825409e+00],
			[3.16865825e-03, 1.49833260e+00, 1.49849874e+00],
			[4.43752342e-04, 1.49975754e+00, 1.49979871e+00],
			[6.79695435e-05, 1.49994487e+00, 1.49998716e+00],
			[2.49620084e-05, 1.49996524e+00, 1.50000979e+00],
			[8.54296115e-07, 1.49998000e+00, 1.50001915e+00],
			[0.00000000e+00, 1.49998000e+00, 1.50002000e+00],
			[0.00000000e+00, 1.49998000e+00, 1.50002000e+00],
		])
		
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, atol=1e-3))

		expected_AB = np.array([0.42808481, 0.42927359, 0.1426416])
		expected_AC = np.array([0.42909702, 0.42790873, 0.14299425])
		self.assertTrue(np.allclose(res.get_last_state_dist("A -- B").values, expected_AB, atol=1e-3))
		self.assertTrue(np.allclose(res.get_last_state_dist("A -- C").values, expected_AC, atol=1e-3))

	def test_log_growth(self):

		sim = PopSimulation(join(dirname(__file__), "pop", "Log_Growth.pbnd"), join(dirname(__file__), "pop", "Log_Growth.cfg"))
		res = sim.run()
		
		expected = np.array([
			[1.24918263],
			[1.74939875],
			[2.24721216],
			[2.74422511],
			[3.24520922],
			[3.74718037],
			[4.24208806],
			[4.73329671],
			[5.23582238],
			[5.75896476],
		])
		
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, atol=1e-5))
		
		expected = np.array([0.008984, 0.039730, 0.096174, 0.151794, 0.197019, 0.179636, 0.110641, 0.104453, 0.053642, 0.036398, 0.011686, 0.004299, 0.002714, 0.002832])
		self.assertTrue(np.allclose(res.get_last_state_dist("A").values, expected, atol=1e-5))