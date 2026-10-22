"""Test suite for uppmaboss simulations of cMaBoSS models (mirrors test/test_uppmaboss.py and test/test_uppmaboss_cmaboss.py).

The models are loaded with cmaboss=True and UpdatePopulation.run() is called without
cmaboss=True: a cMaBoSS model should be simulated with cMaBoSS without asking.
Expected values are the cMaBoSS ones (they differ slightly from the MaBoSS binary).

Workdirs are distinct from the ones of test/test_uppmaboss.py. Overwrite and Restore share
WT2_cmaboss on purpose: Restore reloads the results written by Overwrite (classes run in
alphabetical order).
"""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load, UpdatePopulation
from os.path import dirname, join

path = dirname(dirname(__file__))

expected_pop_ratios_1h = [
	1.0, 0.6817954975654412, 0.6306956613676165, 0.6114502036053414, 0.5974385752516413,
	0.5754904027149489, 0.5431751367072919, 0.5029923192031418, 0.46040464153785243,
	0.4199042244647881, 0.38405999457910445, 0.3541393222851166, 0.3292863788241307
]

expected_pop_ratios_2h = [
	1.0, 0.6291157526378983, 0.5951898920757873, 0.5364762686201768, 0.4518973015533213,
	0.37686818489759966, 0.3252405155383433, 0.2888494879411558, 0.26076163549528764,
	0.238209525180228, 0.21933083001117923, 0.20322239831032154, 0.18952228285642161
]


class TestUpPMaBoSScMaBoSS(TestCase):

	def test_uppmaboss(self):

		sim = load(join(path, "CellFateModel.bnd"), join(path, "CellFateModel_1h.cfg"), cmaboss=True)
		uppmaboss_model = UpdatePopulation(sim, join(path, "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT_cmaboss_load')
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios_1h[i])

	def test_uppmaboss_only_final_state(self):

		sim = load(join(path, "CellFateModel.bnd"), join(path, "CellFateModel_1h.cfg"), cmaboss=True)
		uppmaboss_model = UpdatePopulation(sim, join(path, "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT_cmaboss_load_final', only_final_state=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()
		expected_pop_ratios = [
			1.0, 0.677860000000014, 0.6279830612000052, 0.6097213137802837, 0.5943624338860777,
			0.5680084035675105, 0.5307981730497715, 0.48714533129812415, 0.4427225485370006,
			0.40227984372809306, 0.36770389115960767, 0.3393539211511613, 0.31562968852344475
		]

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])


class TestUpPMaBoSSOverwritecMaBoSS(TestCase):

	def test_uppmaboss_overwrite(self):

		# Now again, but with a modified max_time, and with overwrite
		# Results should be different if overwriting indeed works
		sim = load(join(path, "CellFateModel.bnd"), join(path, "CellFateModel_1h.cfg"), cmaboss=True)
		sim.param["max_time"] = 2

		uppmaboss_model = UpdatePopulation(sim, join(path, "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT2_cmaboss', overwrite=True)
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios_2h[i])


class TestUpPMaBoSSRestorecMaBoSS(TestCase):

	def test_uppmaboss_restore(self):
		# Now again, but with save results (from the overwrite run)
		sim = load(join(path, "CellFateModel.bnd"), join(path, "CellFateModel_1h.cfg"), cmaboss=True)
		uppmaboss_model = UpdatePopulation(sim, join(path, "CellFate_1h.upp"))
		uppmaboss_sim = uppmaboss_model.run('WT2_cmaboss')
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios_2h[i])

		uppmaboss_sim.save('results_cmaboss')
		uppmaboss_sim.results[0].plot_piechart()

		expected_state_tnf = [
			1.e-05, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00,
			0.e+00, 0.e+00, 0.e+00
		]
		for i, val in enumerate(uppmaboss_sim.get_stepwise_probability_distribution().loc[:, "TNF"].values):
			self.assertAlmostEqual(val, expected_state_tnf[i])

		expected_node_tnf = [
			0.7236814615786558, 0.5373134786407424, 0.5114897198560119, 0.5424115736208295,
			0.5556400644694282, 0.563065021116992, 0.5761146424771463, 0.6013790622808339,
			0.6242913912728348, 0.6454884313238968, 0.6674342514854219, 0.6902420730162657,
			0.7104523767707499
		]
		for i, val in enumerate(uppmaboss_sim.get_nodes_stepwise_probability_distribution(nodes=["TNF"]).loc[:, "TNF"].values):
			self.assertAlmostEqual(val, expected_node_tnf[i])

		expected_sum = [
			0.7034538925027969, 0.43769480039312103, 0.3731220329644845, 0.40822930597900997,
			0.42585174224967043, 0.4192040754144225, 0.41192961664130595, 0.41457444020531586,
			0.4144073292759453, 0.41072387254015663, 0.4045702997832173, 0.39629514157665213,
			0.3837682401629574
		]
		for i, val in enumerate(uppmaboss_sim.get_stepwise_probability_distribution(include=["TNF"], exclude=["NFkB"]).sum(axis=1).values):
			self.assertAlmostEqual(val, expected_sum[i])

		following_sim = UpdatePopulation(
			sim,
			join(path, "CellFate_1h.upp"),
			previous_run=uppmaboss_sim
		)
		following_sim.run()


class TestUpPMaBoSSRemotecMaBoSS(TestCase):

	def test_uppmaboss_remote(self):

		sim = load(
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.bnd",
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.cfg",
			cmaboss=True
		)
		uppmaboss_model = UpdatePopulation(sim,
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.upp"
		)
		uppmaboss_sim = uppmaboss_model.run('remote_cmaboss')


class TestUpPMaBoSSWritecMaBoSS(TestCase):
	def test_uppmaboss_write(self):

		sim = load(
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.bnd",
			"https://raw.githubusercontent.com/sysbio-curie/UPMaBoSS-docker/master/CellFate/CellFateModel_uppmaboss.cfg",
			cmaboss=True
		)
		uppmaboss_model = UpdatePopulation(sim)

		uppmaboss_model.setDivisionNode("Division")
		uppmaboss_model.setDeathNode("Death")
		uppmaboss_model.setExternalVariable("$TNF_induc", "$ProdTNF_NFkB*p[(NFkB,Death) = (1,0)]")
		uppmaboss_model.setStepNumber(12)
		uppmaboss_sim = uppmaboss_model.run('defined_cmaboss')

		expected_pop_ratios = [
			1.0, 0.9108545331223944, 0.8426781761741082, 0.8162475495821161, 0.7975658689187353,
			0.7682633478319157, 0.7252321480257132, 0.6718113242190726, 0.6150531749093324,
			0.5612292077323735, 0.5134596731550568, 0.47318778568505027, 0.4397655749228259
		]
		pop_ratios = uppmaboss_sim.get_population_ratios('WT').values.tolist()

		for i, pop_ratio in enumerate(pop_ratios):
			self.assertAlmostEqual(pop_ratio, expected_pop_ratios[i])
