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
			[1.0026182,  0.12352425, 0.00388882, 0.11915136],
			[1.01221041, 0.36764729, 0.03228032, 0.33726072],
			[1.02404134, 0.6148501,  0.07797315, 0.53034757],
			[1.04968372, 0.85749484, 0.12923471, 0.70781184],
			[1.06167131, 1.10236379, 0.1969001,  0.88427401],
			[1.07204255, 1.34929362, 0.27820032, 1.04764388],
			[1.09217474, 1.58483432, 0.36569607, 1.19938293],
			[1.10504707, 1.82428736, 0.45915183, 1.34481045],
			[1.14415503, 2.06206716, 0.55558451, 1.47401568],
			[1.18784968, 2.31569521, 0.66176126, 1.59365861],
		])
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, rtol=5e-2, atol=1e-2))
		
		expected = np.array([0.63652264, 0.25907309, 0.08950699, 0.01300554, 0.00189174])
		self.assertTrue(np.allclose(res.get_last_state_dist("B -- C").values, expected, rtol=5e-2, atol=1e-2))
		
	def test_assymetric_restore(self):

		sim = PopSimulation(join(dirname(__file__), "pop", "Assymetric.pbnd"), join(dirname(__file__), "pop", "Assymetric.cfg"))
		res = sim.run(workdir="pop", prefix="res_assymetric", overwrite=False)

		expected = np.array([
			[1.0026182,  0.12352425, 0.00388882, 0.11915136],
			[1.01221041, 0.36764729, 0.03228032, 0.33726072],
			[1.02404134, 0.6148501,  0.07797315, 0.53034757],
			[1.04968372, 0.85749484, 0.12923471, 0.70781184],
			[1.06167131, 1.10236379, 0.1969001,  0.88427401],
			[1.07204255, 1.34929362, 0.27820032, 1.04764388],
			[1.09217474, 1.58483432, 0.36569607, 1.19938293],
			[1.10504707, 1.82428736, 0.45915183, 1.34481045],
			[1.14415503, 2.06206716, 0.55558451, 1.47401568],
			[1.18784968, 2.31569521, 0.66176126, 1.59365861],
		])
		self.assertTrue(np.allclose(res.get_simple_states_popsize().values, expected, rtol=5e-2, atol=1e-2))
		
		expected = np.array([0.63652264, 0.25907309, 0.08950699, 0.01300554, 0.00189174])
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