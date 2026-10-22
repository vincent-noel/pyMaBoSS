"""Test suite for building observed state transition graph with cMaBoSS (mirrors test/test_observed_graph.py)."""

#For testing in environnement with no screen
import matplotlib
matplotlib.use('Agg')
import pandas, numpy

from unittest import TestCase
from maboss import load
from os.path import dirname, join

path = dirname(dirname(__file__))


class TestObservedGraphcMaBoSS(TestCase):

	def test_cell_cycle(self):

		sim = load(join(path, "cellcycle.bnd"), join(path, "cellcycle_runcfg.cfg"), cmaboss=True)
		sim.update_parameters(use_physrandgen = False)
		sim.network.set_observed_graph_nodes(["CycD", "CycA", "CycE"])
		res = sim.run()

		ref_observed_graph = numpy.array([
			[0.        , 0.        , 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.        ],
			[0.        , 0.        , 0.        , 0.50555257, 0.        ,
				0.49444743, 0.        , 0.        ],
			[0.        , 0.        , 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.        ],
			[0.        , 0.00537443, 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.99462557],
			[0.        , 0.        , 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.        ],
			[0.        , 0.94614513, 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.05385487],
			[0.        , 0.        , 0.        , 0.        , 0.        ,
				0.        , 0.        , 0.        ],
			[0.        , 0.        , 0.        , 0.00715549, 0.        ,
				0.99284451, 0.        , 0.        ]
		])

		self.assertTrue(numpy.isclose(
			res.get_observed_graph().values, ref_observed_graph
		).all())

		res.plot_observed_graph()
