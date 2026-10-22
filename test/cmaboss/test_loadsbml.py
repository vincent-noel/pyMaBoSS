"""Test suite for loading SBML and BNet models with cMaBoSS (mirrors test/test_loadsbml.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import loadBNet, loadSBML
from os.path import dirname, join

path = dirname(dirname(__file__))

class TestLoadSBMLcMaBoSS(TestCase):

	def test_loadbnet(self):

		sim = loadBNet(
			join(path, "ensemble", "TC2_BN_0.bnet"),
			cmaboss=True
		)

		self.assertEqual(
			sorted(list(sim.network.keys())),
			[
				'AHR', 'BCL6', 'CEBPB', 'CTCF', 'E2F3', 'E2F7', 'EBF1', 'EGR3', 'ESRRA', 'ETV5',
				'FOSL1', 'FOSL2', 'FOXM1', 'FOXO3', 'HEY1', 'HIF1A', 'HMGA2', 'HSF1', 'HSF2',
				'KLF15', 'KLF9', 'MAX', 'NFAT5', 'NFATC3', 'NFE2L2', 'NR1H3', 'NR2F1', 'NR2F2',
				'NR3C1', 'PPARG', 'RARA', 'RBPJ', 'RUNX2', 'SMAD3', 'SNAI1', 'SP3', 'STAT5A',
				'TCF12', 'THAP11', 'TP53', 'TP63', 'VDR', 'XBP1', 'YBX1', 'YY1', 'ZNF143'
			]
		)

	def test_load_sbml(self):

		sim = loadSBML(
			join(path, "Cacace_TdevModel_2nov2020.sbml"),
			join(path, "Cacace_TdevModel_2nov2020.cfg"),
			cmaboss=True
		)

		res = sim.run()

		self.assertEqual(
			list(res.get_last_states_probtraj().values), [1.0]
		)

		self.assertEqual(
			list(res.get_last_states_probtraj().columns), ['Id1 -- Flt3']
		)

	def test_load_sbml_corral(self):

		sim = loadSBML(
			join(path, "Corral_ThIL17diff_15jan2021.sbml"),
			cmaboss=True
		)

		sim.update_parameters(sample_count=1000, max_time=10)
		res = sim.run()

		last_states = res.get_last_states_probtraj().iloc[0]
		expected_last_states = {
			'IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1R1 -- STAT5B_b2 -- CXCR4 -- IL2RB -- CGC': 4.9e-05,
			'IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1RAP -- IL1R1 -- CXCR4 -- IL2RB -- CGC': 0.00096,
			'IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1RAP -- IL1R1 -- STAT5B_b2 -- CXCR4 -- IL2RB -- CGC': 0.998991,
		}

		self.assertEqual(sorted(last_states.index), sorted(expected_last_states.keys()))
		for state, proba in expected_last_states.items():
			self.assertAlmostEqual(last_states[state], proba)
