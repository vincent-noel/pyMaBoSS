"""Test suite for loading models."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load, loadBNet, loadSBML
from os.path import dirname, join, exists
import shutil

class TestLoadSBML(TestCase):

	def test_loadbnet(self):

		sim = loadBNet(
			join(dirname(__file__), "ensemble", "TC2_BN_0.bnet")
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
			join(dirname(__file__), "Cacace_TdevModel_2nov2020.sbml"),
			join(dirname(__file__), "Cacace_TdevModel_2nov2020.cfg"),
		)
		
			
		res = sim.run()
		
		self.assertEqual(
			list(res.get_last_states_probtraj().values), [1.0]
		)
		
		self.assertEqual(
			list(res.get_last_states_probtraj().columns), ['Id1 -- Flt3']
		)
		
	def test_load_sbml_cmaboss(self):
		
		ref_logical_rules = {
			'CEBPb': '((!CEBPb & CEBPa) & (Pu1_b1 & Pu1_b2)) | (CEBPb & (Pu1_b1 & Pu1_b2))', 
			'CEBPa': '(((Pu1_b1 & Pu1_b2) & !Foxo1) & Runx1_b1) & !Hes1', 
			'Pu1_b1': '(((((((((((Pu1_b1 & !Pax5) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !TCF1) & (Runx1_b1 & !Runx1_b2)) & !MCSF) | ((((Pu1_b1 & !Pax5) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !TCF1) & (Runx1_b1 & Runx1_b2))) | ((((Pu1_b1 & !Pax5) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & TCF1) & Runx1_b1)) | (((Pu1_b1 & !Pax5) & (Gata3_b1 & Gata3_b2)) & Runx1_b1)) | ((Pu1_b1 & Pax5) & Runx1_b1)) | (((((Pu1_b1 & !Pax5) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !TCF1) & (Runx1_b1 & !Runx1_b2)) & MCSF)) & (!Pu1_b1 | (Pu1_b1 & !Pu1_b2))) | (Pu1_b1 & Pu1_b2)', 
			'Pu1_b2': '(((((Pu1_b1 & !Pax5) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !TCF1) & (Runx1_b1 & !Runx1_b2)) & MCSF) & Pu1_b1', 
			'Foxo1': '((!CEBPa & !EBF1) & E2A_protein) | (!CEBPa & EBF1)', 
			'Runx1_b1': '(((((!Scl & Pu1_b1) | ((Scl & (Pu1_b1 & !Pu1_b2)) & !prog_ass_Eprot)) | (Scl & (Pu1_b1 & Pu1_b2))) | (((Scl & !Pu1_b1) & prog_ass_Eprot) | ((Scl & (Pu1_b1 & !Pu1_b2)) & prog_ass_Eprot))) & (!Runx1_b1 | (Runx1_b1 & !Runx1_b2))) | (Runx1_b1 & Runx1_b2)', 
			'Hes1': '((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & HEB_E2A) & NTC', 'EBF1': '(((((((((((!CEBPb & !CEBPa) & !Pu1_b1) & EBF1) & Pax5) & Foxo1) & E2A_protein) & Ets1) & !Gata3_b1) & !NTC) | (((((((((!CEBPb & !CEBPa) & Pu1_b1) & !EBF1) & Foxo1) & E2A_protein) & Ets1) & !Gata3_b1) & !NTC) & Stat5)) | ((((((((((!CEBPb & !CEBPa) & Pu1_b1) & EBF1) & !Pax5) & Foxo1) & E2A_protein) & Ets1) & !Gata3_b1) & !NTC) & Stat5)) | (((((((((!CEBPb & !CEBPa) & Pu1_b1) & EBF1) & Pax5) & Foxo1) & E2A_protein) & Ets1) & !Gata3_b1) & !NTC)', 
			'HEB_E2A': '((!Id2 & !Id3) & E2A_protein) & HEB', 
			'Bcl11b': '(((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & Gata3_b1) & TCF1) & HEB_E2A) & Runx1_b1) & NTC', 
			'Gfi1': '((!CEBPa & (!Pu1_b1 | (Pu1_b1 & !Pu1_b2))) & HEB_E2A) | (CEBPa & (!Pu1_b1 | (Pu1_b1 & !Pu1_b2)))', 
			'Id2': '(((Pu1_b1 & !EBF1) & !HEB_E2A) & !Bcl11b) & !Gfi1', 'Id1': '!Gfi1', 'Scl': '(((!Gata3_b1 & !HEB_E2A) & !Bcl11b) & Gfi1) | ((Gata3_b1 & !HEB_E2A) & !Bcl11b)', 'Lyl1': 'Pu1_b1 & !NTC', 'prog_ass_Eprot': '((((!Scl & !Lmo2) & Lyl1) & E2A_protein) | ((!Scl & Lmo2) & E2A_protein)) | (Scl & E2A_protein)', 'NTC': 'Notch1_rec & !Deltex', 'Stat5': 'IL7Ra_act', 'Id3': '(((((((((!Scl & !Pu1_b1) & !Lyl1) & prog_ass_Eprot) & !NTC) & Stat5) | ((((!Scl & !Pu1_b1) & Lyl1) & prog_ass_Eprot) & !NTC)) | (((((!Scl & (Pu1_b1 & Pu1_b2)) & !Lyl1) & prog_ass_Eprot) & !NTC) & Stat5)) | ((((!Scl & (Pu1_b1 & Pu1_b2)) & Lyl1) & prog_ass_Eprot) & !NTC)) | (((Scl & !Pu1_b1) & prog_ass_Eprot) & !NTC)) | (((Scl & (Pu1_b1 & Pu1_b2)) & prog_ass_Eprot) & !NTC)', 'Gata3_b1': '(((((((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & !TCF1) & HEB_E2A) & NTC) | ((((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & TCF1) & !HEB_E2A) & Bcl11b) & NTC)) | (((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & TCF1) & HEB_E2A) & NTC)) | ((((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & TCF1) & !HEB_E2A) & !Bcl11b) & NTC)) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) | (Gata3_b1 & Gata3_b2)', 'E2A_protein': '!Id2 & E2A_gene', 'Scl_E2A': '(Scl & E2A_protein) & !HEB_E2A', 'Lmo2': '((Pu1_b1 & Pu1_b2) & !HEB_E2A) & Gfi1', 'Gfi1b': '((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & Gata3_b1) & HEB_E2A) & !Bcl11b) & NTC', 'Kit': '(((((((!Scl & !Lmo2) & prog_ass_Eprot) & !Gata3_b1) & !Bcl11b) & Stat5) | ((((!Scl & !Lmo2) & prog_ass_Eprot) & Gata3_b1) & !Bcl11b)) | (((!Scl & Lmo2) & prog_ass_Eprot) & !Bcl11b)) | ((Scl & prog_ass_Eprot) & !Bcl11b)', 'Flt3': '!Bcl11b', 'Pax5': '!CEBPa & EBF1', 'Gata3_b2': '((((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & TCF1) & !HEB_E2A) & !Bcl11b) & NTC) & Gata3_b1', 'TCF1': '(((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !EBF1) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & NTC', 'Runx1_b2': '(((Scl & !Pu1_b1) & prog_ass_Eprot) | ((Scl & (Pu1_b1 & !Pu1_b2)) & prog_ass_Eprot)) & Runx1_b1', 'MCSF': 'MCSF', 'Hhex': '((!Scl & Lmo2) & (!Pu1_b1 | (Pu1_b1 & !Pu1_b2))) | (Scl & (!Pu1_b1 | (Pu1_b1 & !Pu1_b2)))', 'Ikaros': '(Pu1_b1 & !Pu1_b2) & Runx1_b1', 'Ets1': '((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !E2A_protein) & Gata3_b1) & Bcl11b) | ((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & E2A_protein)', 'Bcl11a': '(Pu1_b1 & Pu1_b2) & !Gata3_b1', 'E2A_gene': '(((((!Pu1_b1 & Ikaros) & EBF1) & Pax5) | ((((Pu1_b1 & !Pu1_b2) & Ikaros) & !EBF1) & NTC)) | (((((Pu1_b1 & !Pu1_b2) & Ikaros) & EBF1) & !Pax5) & NTC)) | ((((Pu1_b1 & !Pu1_b2) & Ikaros) & EBF1) & Pax5)', 'Myb': 'Pu1_b1 & !Pu1_b2', 'CD25': '(TCF1 & NTC) & Stat5', 'Runx3': '(Scl & !Pu1_b1) & prog_ass_Eprot', 'HEB_gene': '(Pu1_b1 & !Pu1_b2) & NTC', 'HEB': 'HEB_gene', 'Lef1_b1': '(((TCF1 & NTC) | (!TCF1 & NTC)) & (!Lef1_b1 | (Lef1_b1 & !Lef1_b2))) | (Lef1_b1 & Lef1_b2)', 'Lef1_b2': '(!TCF1 & NTC) & Lef1_b1', 'pTa': '((((!Scl & !prog_ass_Eprot) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & HEB_E2A) & Bcl11b) & NTC', 'TCRb': '(((Ets1 & Gata3_b1) & HEB_E2A) & Bcl11b) & Runx1_b1', 'Rag1': '(((((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !HEB_E2A) & (!Runx1_b1 | (Runx1_b1 & !Runx1_b2))) & !Gfi1) & NTC) | (((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & !HEB_E2A) & (!Runx1_b1 | (Runx1_b1 & !Runx1_b2))) & Gfi1)) | ((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & (!Gata3_b1 | (Gata3_b1 & !Gata3_b2))) & HEB_E2A) & (!Runx1_b1 | (Runx1_b1 & !Runx1_b2)))', 'CD3e': '(((((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !HEB_E2A) & !Bcl11b) & NTC) | (((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !HEB_E2A) & Bcl11b)) | ((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & HEB_E2A)', 'CD3g': '((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & HEB_E2A) & NTC', 'Lat': '(((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & !HEB_E2A) & NTC) | ((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & HEB_E2A)', 'Zap70': '((!Pu1_b1 | (Pu1_b1 & !Pu1_b2)) & HEB_E2A) & Bcl11b', 'Nrarp': 'NTC', 'Notch_gene_b1': '(((((((!Lyl1 & !Pax5) & HEB_E2A) & NTC) & Nrarp) | (((Lyl1 & !Pax5) & HEB_E2A) & Nrarp)) | (((((!Lyl1 & !Pax5) & HEB_E2A) & NTC) & !Nrarp) | (((Lyl1 & !Pax5) & HEB_E2A) & !Nrarp))) & (!Notch_gene_b1 | (Notch_gene_b1 & !Notch_gene_b2))) | (Notch_gene_b1 & Notch_gene_b2)', 'Notch_gene_b2': '(((((!Lyl1 & !Pax5) & HEB_E2A) & NTC) & !Nrarp) | (((Lyl1 & !Pax5) & HEB_E2A) & !Nrarp)) & Notch_gene_b1', 'Notch1_rec': 'Notch_gene_b1 & Delta', 'Deltex': '!HEB_E2A & NTC', 'Delta': 'Delta', 'IL7Ra_gene': '(!Pu1_b1 & Notch_gene_b1) | Pu1_b1', 'IL7': 'IL7', 'IL7Ra_act': 'IL7Ra_gene & IL7', 'CD45': 'Scl_E2A & (Gata3_b1 & Gata3_b2)', 'CD44': 'Pu1_b1'
		}
		
		sim = loadSBML(
			join(dirname(__file__), "Cacace_TdevModel_2nov2020.sbml"),
			join(dirname(__file__), "Cacace_TdevModel_2nov2020.cfg"),
			cmaboss=True
		)
		
		rules = sim.get_logical_rules()
		for node, rule in rules.items():
			self.assertEqual(rule, ref_logical_rules[node])
		
		
		res = sim.run()
		
		self.assertEqual(
			list(res.get_last_states_probtraj().values), [1.0]
		)
		
		self.assertEqual(
			list(res.get_last_states_probtraj().columns), ['Id1 -- Flt3']
		)
		
	def test_load_sbml_corral(self):
		
		sim = loadSBML(
			join(dirname(__file__), "Corral_ThIL17diff_15jan2021.sbml"),
		)
		
		sim.update_parameters(sample_count=1000, max_time=10)
		res = sim.run()
		
		self.assertEqual(
			list(res.get_last_states_probtraj().values[0]), [4.9e-05, 0.00096, 0.998991]
		)

		self.assertEqual(
			list(res.get_last_states_probtraj().columns.values), 
			['IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1R1 -- STAT5B_b2 -- CXCR4 -- IL2RB -- CGC', 
			'IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1RAP -- IL1R1 -- CXCR4 -- IL2RB -- CGC', 
			'IL12RB1 -- IL12RB2 -- GP130 -- STAT5B_b1 -- IL1RAP -- IL1R1 -- STAT5B_b2 -- CXCR4 -- IL2RB -- CGC'
			]
		)
		
	
		