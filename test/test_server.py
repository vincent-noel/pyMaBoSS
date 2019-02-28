"""Test suite for loading models."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')

from unittest import TestCase
from maboss import load
from maboss.maboss_server import MaBoSSClient
from os.path import dirname, join


class TestServer(TestCase):

	def test_run_p53_Mdm2(self):

		sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
		mbcli = MaBoSSClient(host="localhost", port=7777)
		res = mbcli.run(sim)

		self.assertEqual(
			res.getFP(),
			"Fixed Points (1)\n"
			+ "FP\tProba\tState\tMdm2N\tp53_h\tp53\tMdm2C\tDam\n"
			+ "#1\t0.90548\tMdm2N\t1\t0\t0\t0\t0\n"
		)
		mbcli.close()
