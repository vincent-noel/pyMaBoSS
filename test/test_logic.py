"""Test the functions in logic.py."""

from unittest import TestCase
import sys
sys.path.append('..')
from maboss import logic

class TestLogic(TestCase):

	def test_logic(self):


		print("Check _check_logic_defined")
		print("   zerology")
		self.assertTrue(logic._check_logic_defined([], []))
		self.assertTrue(logic._check_logic_defined([], ["True"]))

		print("   normal case")
		self.assertTrue(logic._check_logic_defined(['foo', 'bar'], ['(foo || bar)']))

		print("   syntax error")
		self.assertTrue(not logic._check_logic_defined(['foo', 'bar'], ['foo bar']))

		print("   undifined variable")
		self.assertTrue(not logic._check_logic_defined(['foo', 'bar'], ['(foo && barr)']))

		print("    light syntax")
		self.assertTrue(logic._check_logic_defined(['a', 'b', 'c', 'd'], ["a & !b AND c OR a & b & !c OR !a & b & c"]))

		print("   conditional expression")
		self.assertTrue(logic._check_logic_defined(['a', 'b', 'c'], ["a ? b : c"]))

		print("   other operators within conditional expression")
		self.assertTrue(logic._check_logic_defined(['a', 'b', 'c', 'd', 'e', 'f', 'g'], ["a & b ^ !c ? d | !e : f ^ g"]))

