"""Test suite for simulation probability distributions with cMaBoSS (mirrors test/test_probtrajs.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')
import pandas, numpy

from unittest import TestCase
from maboss import load, set_nodes_istate
from os.path import dirname, join

path = dirname(dirname(__file__))

def read_ref(name):
	return pandas.read_csv(join(path, "res", name), index_col=0, header=0)


class TestProbTrajscMaBoSS(TestCase):

	def assertFrameClose(self, result, reference, **kwargs):
		result = result.sort_index(axis=1)
		reference = reference.sort_index(axis=1)
		self.assertEqual(list(result.columns), list(reference.columns))
		self.assertEqual(result.shape, reference.shape)
		self.assertTrue(numpy.isclose(result, reference, **kwargs).all())

	def test_probtraj_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		self.assertFrameClose(res.get_states_probtraj(), read_ref("p53_Mdm2_states_probtraj.csv"), atol=1e-5)
		self.assertFrameClose(res.get_last_states_probtraj(), read_ref("p53_Mdm2_last_states_probtraj.csv"), atol=1e-5)
		self.assertFrameClose(res.get_nodes_probtraj(), read_ref("p53_Mdm2_nodes_probtraj.csv"), atol=1e-5)

	def test_probtraj_errors_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		self.assertFrameClose(res.get_states_probtraj_errors(), read_ref("p53_Mdm2_states_probtraj_errors.csv"), atol=1e-5)
		self.assertFrameClose(res.get_nodes_probtraj_error(), read_ref("p53_Mdm2_nodes_probtraj_error.csv"), atol=1e-5)
		self.assertFrameClose(res.get_last_nodes_probtraj(), read_ref("p53_Mdm2_last_nodes_probtraj.csv"), atol=1e-5)

	def test_entropy_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		self.assertFrameClose(res.get_entropy_trajectory(), read_ref("p53_Mdm2_entropy_trajectory.csv"), atol=1e-4)
		self.assertFrameClose(res.get_entropy_trajectory_error(), read_ref("p53_Mdm2_entropy_trajectory_error.csv"), atol=1e-4)

	def test_fptable_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		fptable = res.get_fptable()
		ref_fptable = read_ref("p53_Mdm2_fptable.csv")
		self.assertEqual(sorted(fptable.columns), sorted(ref_fptable.columns))
		self.assertEqual(
			fptable[ref_fptable.columns].astype(str).values.tolist(),
			ref_fptable.astype(str).values.tolist()
		)

	def test_probtraj_cellcycle(self):

		sim = load(
			join(path, "cellcycle.bnd"),
			join(path, "cellcycle_runcfg.cfg"),
			join(path, "cellcycle_runcfg-thread_1.cfg"),
			cmaboss=True
		)
		res = sim.run()

		self.assertFrameClose(res.get_states_probtraj(), read_ref("cellcycle_states_probtraj.csv"))
		self.assertFrameClose(res.get_last_states_probtraj(), read_ref("cellcycle_last_states_probtraj.csv"))
		self.assertFrameClose(res.get_nodes_probtraj(), read_ref("cellcycle_nodes_probtraj.csv"))

	def test_toyexample(self):

		sim = load(join(path, "Four_cycle.bnd"), join(path, "Four_cycle_FEscape.cfg"), cmaboss=True)
		set_nodes_istate(sim, ["A", "B", "C"], [1, 0])
		res = sim.run()
		self.assertEqual(
			sorted(res.get_nodes_probtraj().columns.values),
			['A', 'B', 'C']
		)

		sim.network.set_output(['A', 'B'])
		res = sim.run()
		self.assertEqual(
			sorted(res.get_nodes_probtraj().columns.values),
			['A', 'B']
		)

		with self.assertRaises(AssertionError, msg="Node(s) 'D' not defined !"):
			sim.network.set_output(['A', 'B', 'C', 'E'])
