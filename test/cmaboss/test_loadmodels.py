"""Test suite for loading models with cMaBoSS (mirrors test/test_loadmodels.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load
from maboss.cmabosssimulation import CMaBoSSSimulation
from os.path import dirname, join, exists
import shutil

path = dirname(dirname(__file__))

class TestLoadModelscMaBoSS(TestCase):

	def test_load_returns_cmaboss(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		self.assertIsInstance(sim, CMaBoSSSimulation)

	def test_load_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		self.assertEqual(sim.network['Mdm2C'].logExp, "$case_a ? p53_h : p53")

	def test_simulate_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		res.plot_fixpoint()
		res.plot_trajectory(error=True)
		res.plot_node_trajectory(error=True)
		res.plot_entropy_trajectory()
		res.plot_piechart()

	def test_copy_p53_Mdm2(self):
		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		sim_copy = sim.copy()

		self.assertEqual(sim.str_bnd(), sim_copy.str_bnd())
		self.assertEqual(sim.str_cfg(), sim_copy.str_cfg())

		# The copy must be independent from the original
		sim_copy.update_parameters(sample_count=100)
		self.assertNotEqual(sim.param['sample_count'], sim_copy.param['sample_count'])

	def test_check_p53_Mdm2(self):
		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		errors = sim.check()
		self.assertEqual(len(errors), 0)

	def test_modifications_p53_Mdm2(self):
		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		sim.update_parameters(sample_count=100)
		sim.network.set_output(['Mdm2C', 'Mdm2N'])
		sim.mutate("Dam", "ON")
		res = sim.run()
		probas = res.get_last_states_probtraj().iloc[0]

		expected_probas = {
			'<nil>': 0.442395, 'Mdm2C': 0.018827, 'Mdm2N': 0.157108, 'Mdm2N -- Mdm2C': 0.381669
		}

		# Expected values come from the MaBoSS binary, which rounds them to 6 decimals
		self.assertEqual(sorted(probas.index), sorted(expected_probas.keys()))
		for state, proba in expected_probas.items():
			self.assertAlmostEqual(probas[state], proba, delta=1e-6)

		if exists("saved_sim_cmaboss"):
			shutil.rmtree("saved_sim_cmaboss")
		res.save("saved_sim_cmaboss")
		self.assertTrue(exists("saved_sim_cmaboss"))
		self.assertTrue(exists("saved_sim_cmaboss/saved_sim_cmaboss.bnd"))
		self.assertTrue(exists("saved_sim_cmaboss/saved_sim_cmaboss.cfg"))
		self.assertTrue(exists("saved_sim_cmaboss/res_probtraj.csv"))
		self.assertTrue(exists("saved_sim_cmaboss/res_fp.csv"))
		self.assertTrue(exists("saved_sim_cmaboss/res_statdist.csv"))

	def test_load_multiple_cfgs(self):

		sim = load(join(path, "reprod_all.bnd"), cmaboss=True)

		sim2 = load(
			join(path, "p53_Mdm2.bnd"),
			join(path, "p53_Mdm2_runcfg.cfg"),
			cmaboss=True
		)

		sim3 = load(
			join(path, "cellcycle.bnd"),
			join(path, "cellcycle_runcfg.cfg"),
			join(path, "cellcycle_runcfg-thread_1.cfg"),
			cmaboss=True
		)
		# The second cfg sets thread_count = 1
		self.assertEqual(sim3.param['thread_count'], 1)

	def test_type_istate(self):

		sim = load(
			join(path, "TregModel_InitPop.bnd"),
			join(path, "TregModel_InitPop_ActTCR2_TGFB.cfg"),
			cmaboss=True
		)

		istate = sim.network.get_istate()

		self.assertEqual([type(value) for value in istate["PTEN"].values()], [float, float])
		self.assertEqual([type(value) for value in istate[("TCR_b1", "TCR_b2", "CD28")].values()], [str, str, str])
		self.assertEqual([type(value) for value in istate[("PI3K_b1", "PI3K_b2")].values()], [float, float, float])
		self.assertEqual([type(value) for value in istate["TGFB"].values()], [str, str])

	def test_load_confusedparser(self):

		sim = load(
			join(path, "confused_parser.bnd"),
			join(path, "confused_parser.cfg"),
			cmaboss=True
		)

		self.assertEqual(
			list(sim.network.keys()),
			['A', 'NOTH', 'B']
		)

	def test_load_reserved_names(self):

		# The exact message comes from the MaBoSS parser, not from pyMaBoSS
		with self.assertRaises(Exception) as context:
			sim = load(
				join(path, "reserved_names.bnd"),
				join(path, "reserved_names.cfg"),
				cmaboss=True
			)
		self.assertIn("NOT", str(context.exception))
