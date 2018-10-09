"""Test suite for loading models."""


from maboss import load
from os.path import dirname, join

sim = load(join(dirname(__file__), "p53_Mdm2.bnd"), join(dirname(__file__), "p53_Mdm2_runcfg.cfg"))
assert(sim.network['Mdm2C'].logExp == "$case_a ? p53_h : p53")
