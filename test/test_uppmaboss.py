"""Test suite for uppmaboss simulations."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load, UpdatePopulation
from os.path import dirname, join


class TestUpPMaBoSS(TestCase):

	def test_uppmaboss(self):

		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"))
		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT')
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()
		expected_pop_ratios = [
			1.0, 0.6817469999999883, 0.6307373259659714, 0.611569849367168, 0.5975759080738972, 
			0.5761277135812584, 0.5432025908777551, 0.5028866377853486, 0.460771390530693, 
			0.4201875679954881, 0.38478088276588096, 0.35500692283830454, 0.3300712365781111
		]

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])

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

class TestUpPMaBoSSOverwrite(TestCase):

	def test_uppmaboss_overwrite(self):

		# Now again, but with a modified max_time, and with overwrite
		# Results should be different if overwriting indeed works
		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"))
		sim.param["max_time"] = 2
		expected_pop_ratios = [
			1.0, 0.6290939999999912, 0.5951210367179476, 0.5367509703155656, 0.45236674001930355, 
			0.3774244993004896, 0.3253316150580102, 0.28897222842748704, 0.26132018691701203, 
			0.23839378295820296, 0.21932847855988485, 0.2033278080634895, 0.18987482697076252
		]

		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT', overwrite=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])


class TestUpPMaBoSSRestore(TestCase):

	def test_uppmaboss_restore(self):
		# Now again, but with save results (from the overwrite run)
		expected_pop_ratios = [
			1.0, 0.6290939999999912, 0.5951210367179476, 0.5367509703155656, 0.45236674001930355, 
			0.3774244993004896, 0.3253316150580102, 0.28897222842748704, 0.26132018691701203, 
			0.23839378295820296, 0.21932847855988485, 0.2033278080634895, 0.18987482697076252
		]

		sim = load(join(dirname(__file__), "CellFateModel.bnd"), join(dirname(__file__), "CellFateModel_1h.cfg"))
		uppmaboss_model = UpdatePopulation(sim, join(dirname(__file__), "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT')
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])

		uppmaboss_sim.save('results')
		uppmaboss_sim.results[0].plot_piechart()

		expected_state_tnf = [
			1.e-05, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
			0.e+00, 0.e+00, 0.e+00
		]
		for i, val in enumerate(uppmaboss_sim.get_stepwise_probability_distribution().loc[:, "TNF"].values):
			self.assertAlmostEqual(val, expected_state_tnf[i])
		
		expected_node_tnf = [
			0.723662, 0.537241, 0.510468, 0.541444, 0.554251, 0.560845, 0.576063, 0.599058,
 			0.623224, 0.645819, 0.668841, 0.691688, 0.710075
		]
		for i, val in enumerate(uppmaboss_sim.get_nodes_stepwise_probability_distribution(nodes=["TNF"]).loc[:, "TNF"].values):
			self.assertAlmostEqual(val, expected_node_tnf[i])

		expected_sum = [
			0.703454, 0.437659, 0.372623, 0.408045, 0.424512, 0.417973, 0.412804, 0.413508,
			0.413859, 0.410776, 0.405925, 0.397581, 0.383826	
		]
		for i, val in enumerate(uppmaboss_sim.get_stepwise_probability_distribution(include=["TNF"], exclude=["NFkB"]).sum(axis=1).values):
			self.assertAlmostEqual(val, expected_sum[i])

		following_sim = UpdatePopulation(
			sim, 
			join(dirname(__file__), "CellFate_1h.upp"), 
			previous_run=uppmaboss_sim
		)
		following_sim.run()


class TestUpPMaBoSSRemote(TestCase):

	def test_uppmaboss_remote(self):

		sim = load(
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.bnd",
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.cfg"
		)
		uppmaboss_model = UpdatePopulation(sim, 
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.upp"
		)
		uppmaboss_sim = uppmaboss_model.run('remote')
	
class TestUpPMaBoSSWrite(TestCase):
	def test_uppmaboss_write(self):

		sim = load(
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.bnd",
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.cfg"
		)
		uppmaboss_model = UpdatePopulation(sim)

		uppmaboss_model.setDivisionNode("Division")
		uppmaboss_model.setDeathNode("Death")
		uppmaboss_model.setExternalVariable("$TNF_induc", "$ProdTNF_NFkB*p[(NFkB,Death) = (1,0)]")
		uppmaboss_model.setStepNumber(12)
		uppmaboss_sim = uppmaboss_model.run('defined')

		expected_pop_ratios = [
			1.0, 0.910830999999983, 0.8425787898459595, 0.8164596899394915, 0.7977586807413645, 
			0.7684801393994071, 0.72538607862224, 0.670962537301388, 0.6143795955681636, 
			0.5607915641038679, 0.5131691444801187, 0.4730567651326412, 0.4394848726246694
		]
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])


