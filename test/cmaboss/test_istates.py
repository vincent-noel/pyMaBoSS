"""Regression tests for oneIstate cfg parsing with cMaBoSS (mirrors test/test_istates.py)."""

from os.path import dirname, join
from unittest import TestCase

from maboss import load

path = dirname(dirname(__file__))


class TestOneIstateParsercMaBoSS(TestCase):
    def test_parse_scalar_istate_false(self):
        sim = load(
            join(path, "test_istates.bnd"),
            join(path, "test_istates.cfg"),
            cmaboss=True
        )

        istate_a = sim.network.get_istate()["A"]
        self.assertTrue(istate_a[0] == 1 and istate_a[1] == 0)

        istate_b = sim.network.get_istate()["B"]
        self.assertTrue(istate_b[0] == 1 and istate_b[1] == 0)

        istate_c = sim.network.get_istate()["C"]
        self.assertTrue(istate_c[0] == 1 and istate_c[1] == 0)

        istate_d = sim.network.get_istate()["D"]
        self.assertTrue(istate_d[0] == 1 and istate_d[1] == 0)

        istate_e = sim.network.get_istate()["E"]
        self.assertTrue(istate_e[0] == 0.5 and istate_e[1] == 0.5)
