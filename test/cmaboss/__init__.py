"""Parity tests for the cMaBoSS implementation.

Each module mirrors a test module of test/, but loads its models with
load(..., cmaboss=True) (and loadBNet/loadSBML/loadTabularQual with cmaboss=True),
so that the cMaBoSS classes can replace Simulation once these all pass.
"""
