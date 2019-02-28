"""Test suite for loading models."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load
from os.path import dirname, join


class TestLoadModels(TestCase):

	def test_load_p53_Mdm2(self):

		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		self.assertEqual(sim.network['Mdm2C'].logExp, "$case_a ? p53_h : p53")

	def test_simulate_p53_Mdm2(self):

		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		res = sim.run()

		res.plot_fixpoint()
		res.plot_node_trajectory()
		res.plot_piechart()

	def test_copy_p53_Mdm2(self):
		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		sim_copy = sim.copy()

		self.assertEqual(str(sim.network), str(sim_copy.network))
		self.assertEqual(sim.str_cfg(), sim_copy.str_cfg())

	def test_check_p53_Mdm2(self):
		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		errors = sim.check()
		self.assertEqual(len(errors), 0)

