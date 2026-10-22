"""Test suite for simulation stationnary distributions with cMaBoSS (mirrors test/test_statdist.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')
import pandas, numpy

from unittest import TestCase
from maboss import load
from os.path import dirname, join

path = dirname(dirname(__file__))


class TestStatDistcMaBoSS(TestCase):

	def test_statdist_p53_Mdm2(self):

		sim = load(join(path, "p53_Mdm2.bnd"), join(path, "p53_Mdm2_runcfg.cfg"), cmaboss=True)
		res = sim.run()

		res_states_statdist = pandas.read_csv(
			join(path, "res", "p53_Mdm2_states_statdist.csv"), index_col=0, header=0
		)

		self.assertTrue(numpy.isclose(
			res.get_states_statdist().sort_index(axis=1).dropna(),
			res_states_statdist.sort_index(axis=1).dropna()
		).all())

		res_statdist_cluster0 = pandas.read_csv(
			join(path, "res", "p53_Mdm2_statdist_cluster0.csv"), index_col=0, header=0
		)

		self.assertTrue(numpy.isclose(
			res.get_statdist_clusters()[0].sort_index(axis=1),
			res_statdist_cluster0.sort_index(axis=1)
		).all())

		res_statdist_clusters_summary = pandas.read_csv(
			join(path, "res", "p53_Mdm2_statdist_clusters_summary.csv"), index_col=0, header=0
		)

		self.assertTrue(numpy.isclose(
			res.get_statdist_clusters_summary().sort_index(axis=1),
			res_statdist_clusters_summary.sort_index(axis=1)
		).all())

		res_statdist_clusters_summary_error = pandas.read_csv(
			join(path, "res", "p53_Mdm2_statdist_clusters_summary_error.csv"), index_col=0, header=0
		)

		self.assertTrue(numpy.isclose(
			res.get_statdist_clusters_summary_error().sort_index(axis=1),
			res_statdist_clusters_summary_error.sort_index(axis=1)
		).all())
