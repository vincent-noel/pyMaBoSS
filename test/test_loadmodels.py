"""Test suite for loading models."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load, loadBNet, loadSBML
from os.path import dirname, join, exists
import shutil

class TestLoadModels(TestCase):

	def test_load_p53_Mdm2(self):

		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		self.assertEqual(sim.network['Mdm2C'].logExp, "$case_a ? p53_h : p53")

	def test_simulate_p53_Mdm2(self):

		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		res = sim.run()

		res.plot_fixpoint()
		res.plot_trajectory(error=True)
		res.plot_node_trajectory(error=True)
		res.plot_entropy_trajectory()
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

	def test_modifications_p53_Mdm2(self):
		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		sim.update_parameters(sample_count=100)
		sim.network.set_output(['Mdm2C', 'Mdm2N'])
		sim.mutate("Dam", "ON")
		res = sim.run()
		probas = res.get_last_states_probtraj().values[0]

		expected_probas = [0.442395, 0.018827, 0.157108, 0.381669]

		for i, proba in enumerate(probas):
			self.assertAlmostEqual(proba, expected_probas[i], delta=proba*1e-6)

		if exists("saved_sim"):
			shutil.rmtree("saved_sim")
		res.save("saved_sim")
		self.assertTrue(exists("saved_sim"))
		self.assertTrue(exists("saved_sim/saved_sim.bnd"))
		self.assertTrue(exists("saved_sim/saved_sim.cfg"))
		self.assertTrue(exists("saved_sim/res_probtraj.csv"))
		self.assertTrue(exists("saved_sim/res_fp.csv"))
		self.assertTrue(exists("saved_sim/res_statdist.csv"))

	def test_load_multiple_cfgs(self):

		sim = load(join(dirname(__file__), "reprod_all.bnd"))

		sim2 = load(
			join(dirname(__file__), "p53_Mdm2.bnd"), 
			join(dirname(__file__), "p53_Mdm2_runcfg.cfg")
		)

		sim3 = load(
			join(dirname(__file__), "cellcycle.bnd"),
			join(dirname(__file__), "cellcycle_runcfg.cfg"),
			join(dirname(__file__), "cellcycle_runcfg-thread_1.cfg")
		)

	def test_type_istate(self):

		sim = load(
			join(dirname(__file__), "TregModel_InitPop.bnd"),
			join(dirname(__file__), "TregModel_InitPop_ActTCR2_TGFB.cfg")
		)

		istate = sim.network.get_istate()

		self.assertEqual([type(value) for value in istate["PTEN"].values()], [float, float])
		self.assertEqual([type(value) for value in istate[("TCR_b1", "TCR_b2", "CD28")].values()], [str, str, str])
		self.assertEqual([type(value) for value in istate[("PI3K_b1", "PI3K_b2")].values()], [float, float, float])
		self.assertEqual([type(value) for value in istate["TGFB"].values()], [str, str])
	
	def test_load_confusedparser(self):
		
		sim = load(
			join(dirname(__file__), "confused_parser.bnd"),
			join(dirname(__file__), "confused_parser.cfg") 
		)
		
		self.assertEqual(
			list(sim.network.keys()),
			['A', 'NOTH', 'B']
		)
		
	def test_load_reserved_names(self):

		with self.assertRaises(Exception) as context:
			sim = load(
				join(dirname(__file__), "reserved_names.bnd"),
				join(dirname(__file__), "reserved_names.cfg") 
			)
		# print(context.exception)
		self.assertTrue(
			   "Name NOT is reserved !" == str(context.exception)
		)

	
		