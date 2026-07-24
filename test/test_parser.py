import pandas as pd

from maboss.temporal_logic.logical_expression_compute import ComputeLogicalExpression
from maboss.temporal_logic.temporal_parser import *
from maboss.temporal_logic.formulas import *
from maboss.temporal_logic.custom_exceptions import *
from unittest import TestCase

# Parser test queries
QUERY = "P(node:name) <= 0.623"
QUERY_INTERROGATION = "T(state:name) = ? [ !A & B | C ]"
QUERY_MULTIPLE_NAMES = "P(node:name,name2,name3) <= 0.5 "
QUERY_INTRICATE_CONDITION = "P(node:name) > 0.5 [ B | ( C & D ) ]"
QUERY_FOUND_BY_NAME = "P(node:AKT1) > 0.623"
QUERY_MULTIPLE_NAMES_AND_CONDITION = "P(node:A,B,C) <= 0.4 [ A | !B & C ]"
QUERY_MULTIPLE_NAMES_AND_INTRICATE_CONDITION = "P(node:A,B,C) >= 0.5 [ B | ( C & D ) ]"
QUERY_MUTATION_CONSTRAINT = "P(node:name) <= 0.5 [ ] [ AKT:ON ]"
QUERY_INCREASE = "Inc(state:name) / [ A & B ] [ AKT:ON ]"
QUERY_INCREASE_NO_LOGICAL = "Inc(state:name) / [ ] [ AKT:ON ]"
QUERY_DECREASE_MULTIPLE_MUTATIONS = "Dec(state:name) / [ A & B ] [ AKT:ON AKT2:OFF BRAF:ON ]"
QUERY_TEST = "Inc(node:Proliferation_b1,Proliferation_b2) / [ ] [ PI3K:ON BRAF:ON ]"

# FORMULA CHECKER test Formulas
NO_ERROR_FORMULA_PROBA = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "1", [], [], [], QUERY)
#NO_ERROR_FORMULA_TIME = Formula(QueryType.T, TargetType.STATE , ["name"], Operators.EQ, "?", ['A','&','B'], [],"T(state:name) = ?")
NO_ERROR_INTERROGATION = Formula(QueryType.P, TargetType.STATE , ["name"], Operators.EQ, "?", ['!A','&','B','|','C'], [], [], "P(state:name) = ? [ !A & B | C ]")
NO_ERROR_INCREASE = Formula(QueryType.INCREASE, TargetType.STATE, ["name"], Operators.NONE, "", [], [['AKT','ON']], [], "Inc(state:name) [ ] [ AKT:ON ]")
NO_ERROR_DECREASE = Formula(QueryType.DECREASE, TargetType.FIXPOINT, ["name"], Operators.NONE, "", ['A','&','B'], [['AKT','ON']], [], "Dec(fp:name) [ A & B ] [ AKT:ON ]")

ERROR_DECREASE_NO_MUTATION = Formula(QueryType.DECREASE, TargetType.FIXPOINT, ["name"], Operators.NONE, "", ['A','&','B'], [], [], "Dec(fp:name) [ A & B ]")
ERROR_INCREASE_NO_MUTATION = Formula(QueryType.INCREASE, TargetType.FIXPOINT, ["name"], Operators.NONE, "", ['A','&','B'], [], [], "Inc(fp:name) [ A & B ]")
ERROR_DECREASE_HAS_OPERATOR = Formula(QueryType.DECREASE, TargetType.FIXPOINT, ["name"], Operators.LE, "0.5", ['A','&','B'], [['AKT','OFF']], [], "Dec(fp:name) [ A & B ] <= 0.5")
ERROR_INCREASE_MULTIPLE_NAMES = Formula(QueryType.INCREASE, TargetType.STATE, ["name","name2"], Operators.NONE, "", ['A','&','B'], [['AKT', 'OFF']], [],"Inc(state:name,name2) [ A & B ] [ AKT:OFF ]")

ERROR_FORMULA_VALUE_EMPTY = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "", [],[],[], QUERY)
ERROR_FORMULA_VALUE_WRONG_SYMBOL = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "!", [],[], [], QUERY)
ERROR_FORMULA_VALUE_NOT_A_NUMBER = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "a", [], [], [],QUERY)

ERROR_FORMULA_TARGET_EMPTY_TAB = Formula(QueryType.P, TargetType.NODE , [], Operators.LE, "0.567", [],[], [], QUERY)
ERROR_FORMULA_TARGET_EMPTY_NAME = Formula(QueryType.P, TargetType.NODE , [""], Operators.LE, "0.567", [],[], [], QUERY)
ERROR_FORMULA_TARGET_EMPTY_MULTIPLE_NAMES = Formula(QueryType.P, TargetType.NODE , ["name","","name2"], Operators.LE, "0.567", [], [] , [], QUERY)

ERROR_FORMULA_INTERROGATION_GRAMMAR =  Formula(QueryType.T, TargetType.STATE , ["name"], Operators.LE, "?", [], [], [], "T(state:name) = ? [ !A & B | C ]")
ERROR_FORMULA_INTERROGATION_GRAMMAR_NO_CONDITIONS = Formula(QueryType.T, TargetType.STATE , ["name"], Operators.EQ, "?", [], [], [], "T(state:name) = ?")

ERROR_VALUE_OVER_ONE_FOR_PROBA = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "1.1", [], [], [], QUERY)
ERROR_VALUE_UNDER_ZERO = Formula(QueryType.P, TargetType.NODE , ["name"], Operators.LE, "-0.1", [], [], [], QUERY)

QUERY_ERROR_TYPE = "V(node:name) > 0.5 [] []"
QUERY_ERROR_OPERATOR = "P(node:name) !< 0.5 [] []"
QUERY_ERROR_TARGET = "P(traj:name) <= 0.5 [] []"

LOGICAL_EXPRESSION_SIMPLE_NO_ERROR = ['A','&', '!B']
LOGICAL_EXPRESSION_INTRICATE_NO_ERROR = ['A','|', ['B','&','C']]

LOGICAL_EXPRESSION_ERROR_NODES = ['A','|', 'B','&','C','D']
LOGICAL_EXPRESSION_ERROR_SYMBOL = ['A', '|' , '&', 'B']
LOGICAL_EXPRESSION_ERROR_FIRST_MEMBER_SYMBOL = ['&','A', '|', 'B']
LOGICAL_EXPRESSION_ERROR_LAST_MEMBER_SYMBOL = ['A', '|', '&', 'B', '|']
LOGICAL_EXPRESSION_ERROR_NODES_INTRICATE = ['A','|', ['B','&','C','D']]


# -------------------------- TESTS PARSER --------------------------------------

class TestParser(TestCase):
    def test_parse_simple_query(self):
        formula1 = Parser.parse_query(QUERY)
        self.assertTrue(formula1.type == QueryType.P)
        self.assertTrue(formula1.operator == Operators.LE)
        self.assertTrue(formula1.value == str(0.623))
        self.assertTrue(formula1.target_name == ['name'])
        self.assertTrue(formula1.logical_equation == [])
        self.assertTrue(formula1.mutation_constraint == [])

    def test_parse_interrogation_query_one_name(self):
        formula2 = Parser.parse_query(QUERY_INTERROGATION)
        #print(formula2)
        self.assertTrue(formula2.type == QueryType.T)
        self.assertTrue(formula2.target == TargetType.STATE)
        self.assertTrue(formula2.operator == Operators.EQ)
        self.assertTrue(formula2.value == "?")
        self.assertTrue(formula2.target_name == ['name'])
        self.assertTrue(formula2.logical_equation == ['!A', '&', 'B', '|', 'C'])
        self.assertTrue(formula2.mutation_constraint == [])

    def test_parse_multiple_names(self):
        formula3 = Parser.parse_query(QUERY_MULTIPLE_NAMES)
        self.assertTrue(formula3.type == QueryType.P)
        self.assertTrue(formula3.operator == Operators.LE)
        self.assertTrue(formula3.value == str(0.5))
        self.assertTrue(formula3.target_name == ['name', 'name2', 'name3'])
        self.assertTrue(formula3.logical_equation == [])
        self.assertTrue(formula3.mutation_constraint == [])

    def test_parse_intricate_condition(self):
        formula4 = Parser.parse_query(QUERY_INTRICATE_CONDITION)
        self.assertTrue(formula4.type == QueryType.P)
        self.assertTrue(formula4.operator == Operators.GT)
        self.assertTrue(formula4.value == str(0.5))
        self.assertTrue(formula4.target_name == ['name'])
        self.assertTrue(formula4.logical_equation == ['B', '|', ['C', '&', 'D']])
        self.assertTrue(formula4.mutation_constraint == [])

    def test_parse_multiple_names_and_condition(self):
        formula5 = Parser.parse_query(QUERY_MULTIPLE_NAMES_AND_CONDITION)
        self.assertTrue(formula5.type == QueryType.P)
        self.assertTrue(formula5.operator == Operators.LE)
        self.assertTrue(formula5.value == str(0.4))
        self.assertTrue(formula5.target_name == ['A', 'B', 'C'])
        self.assertTrue(formula5.logical_equation == ['A', '|', '!B', '&', 'C'])
        self.assertTrue(formula5.mutation_constraint == [])

    def test_parse_multiple_names_and_intricate_condition(self):
        formula6 = Parser.parse_query(QUERY_MULTIPLE_NAMES_AND_INTRICATE_CONDITION)
        self.assertTrue(formula6.type == QueryType.P)
        self.assertTrue(formula6.operator == Operators.GE)
        self.assertTrue(formula6.value == str(0.5))
        self.assertTrue(formula6.target_name == ['A', 'B', 'C'])
        self.assertTrue(formula6.logical_equation == ['B', '|', ['C', '&', 'D']])
        self.assertTrue(formula6.mutation_constraint == [])

    def test_main_query_parser_check_values(self):
        self.assertRaises(ValueError, Parser.parse_query, QUERY_ERROR_TYPE)
        self.assertRaises(ValueError, Parser.parse_query, QUERY_ERROR_OPERATOR)
        self.assertRaises(ValueError, Parser.parse_query, QUERY_ERROR_TARGET)

    def test_increase_parser(self):
        formula7 = Parser.parse_query(QUERY_INCREASE)
        self.assertTrue(formula7.type == QueryType.INCREASE)
        self.assertTrue(formula7.operator == Operators.NONE)
        self.assertTrue(formula7.value == "")
        self.assertTrue(formula7.target_name == ['name'])
        self.assertTrue(formula7.logical_equation == ['A', '&', 'B'])
        self.assertTrue(formula7.mutation_constraint == [['AKT', 'ON']])

    def test_increase_parser_no_logical(self):
        formula8 = Parser.parse_query(QUERY_INCREASE_NO_LOGICAL)
        self.assertTrue(formula8.type == QueryType.INCREASE)
        self.assertTrue(formula8.operator == Operators.NONE)
        self.assertTrue(formula8.value == "")
        self.assertTrue(formula8.target_name == ['name'])
        self.assertTrue(formula8.logical_equation == [])
        self.assertTrue(formula8.mutation_constraint == [['AKT', 'ON']])

    def test_multiple_mutations(self):
        formula9 = Parser.parse_query(QUERY_DECREASE_MULTIPLE_MUTATIONS)
        self.assertTrue(formula9.type == QueryType.DECREASE)
        self.assertTrue(formula9.operator == Operators.NONE)
        self.assertTrue(formula9.value == "")
        self.assertTrue(formula9.target_name == ['name'])
        self.assertTrue(formula9.logical_equation == ['A', '&', 'B'])
        self.assertTrue(formula9.mutation_constraint == [['AKT', 'ON'], ['AKT2', 'OFF'], ['BRAF', 'ON']])
# -------------------------------- FORMULA CHECKER TESTS -------------------------------------
    def test_formula_checker_no_error(self):
        try:
            FormulaChecker.check_formula(NO_ERROR_FORMULA_PROBA)
        except FormulaException:
            self.fail("FormulaChecker.check_formula() PROBA raised an unexpected ValueError")

        try:
            FormulaChecker.check_formula(NO_ERROR_INTERROGATION)
        except FormulaException:
            self.fail("FormulaChecker.check_formula() INTERROGATION raised an unexpected ValueError")

        try:
            FormulaChecker.check_formula(NO_ERROR_INCREASE)
        except FormulaException:
            self.fail("FormulaChecker.check_formula() INCREASE raised an unexpected ValueError")

        try:
            FormulaChecker.check_formula(NO_ERROR_DECREASE)
        except FormulaException:
            self.fail("FormulaChecker.check_formula() DECREASE raised an unexpected ValueError")

    def test_formula_checker_error_value(self):
        self.assertRaises(EmptyValueException, FormulaChecker.check_formula, ERROR_FORMULA_VALUE_EMPTY)
        self.assertRaises(WrongSymbolForValue, FormulaChecker.check_formula, ERROR_FORMULA_VALUE_WRONG_SYMBOL)
        self.assertRaises(WrongSymbolForValue, FormulaChecker.check_formula, ERROR_FORMULA_VALUE_NOT_A_NUMBER)

    def test_formula_checker_error_target_name_empty(self):
        self.assertRaises(EmptyNameException, FormulaChecker.check_formula, ERROR_FORMULA_TARGET_EMPTY_TAB)
        self.assertRaises(EmptyNameException, FormulaChecker.check_formula, ERROR_FORMULA_TARGET_EMPTY_NAME)
        self.assertRaises(EmptyNameException, FormulaChecker.check_formula, ERROR_FORMULA_TARGET_EMPTY_MULTIPLE_NAMES)

    def test_formula_grammar(self):
        self.assertRaises(WrongGrammarException, FormulaChecker.check_formula, ERROR_FORMULA_INTERROGATION_GRAMMAR)
        self.assertRaises(WrongGrammarException, FormulaChecker.check_formula, ERROR_FORMULA_INTERROGATION_GRAMMAR_NO_CONDITIONS)

    def test_formula_value_strict_pos(self):
        self.assertRaises(WrongValueAccordingToType, FormulaChecker.check_formula, ERROR_VALUE_OVER_ONE_FOR_PROBA)
        self.assertRaises(ValueError, FormulaChecker.check_formula, ERROR_VALUE_UNDER_ZERO)

    def test_formula_increase_no_mutation(self):
        self.assertRaises(ErrorInIncreaseDecreaseEvaluation, FormulaChecker.check_formula, ERROR_INCREASE_NO_MUTATION)
        self.assertRaises(ErrorInIncreaseDecreaseEvaluation, FormulaChecker.check_formula, ERROR_DECREASE_NO_MUTATION)

    def test_formula_increase_has_operator(self):
        self.assertRaises(WrongGrammarException, FormulaChecker.check_formula, ERROR_DECREASE_HAS_OPERATOR)

    def test_formula_raise_fixpoint_without_comb(self):
        self.assertRaises(FormulaException, FormulaChecker.check_formula,  Parser.parse_query("P(fp:AKT1,AKT2) >= 0.2 [ ] [ ] [ ]"))

# ---------------------------- TESTS FOR COMPUTATION OF LOGICAL EXPRESSION -----------------------
    def test_expression_no_error(self):
        try:
            ComputeLogicalExpression.check_logical_expression(LOGICAL_EXPRESSION_SIMPLE_NO_ERROR)
        except ErrorInLogicalExpression:
            self.fail("ComputeLogicalExpression.check_logical_expression() raised an unexpected ValueError on SIMPLE")

        try:
            ComputeLogicalExpression.check_logical_expression(LOGICAL_EXPRESSION_INTRICATE_NO_ERROR)
        except ErrorInLogicalExpression:
            self.fail("ComputeLogicalExpression.check_logical_expression() raised an unexpected ValueError on INTRICATE")

    def test_expression_error(self):
        self.assertRaises(ErrorInLogicalExpression, ComputeLogicalExpression.check_logical_expression, LOGICAL_EXPRESSION_ERROR_NODES)
        self.assertRaises(ErrorInLogicalExpression, ComputeLogicalExpression.check_logical_expression, LOGICAL_EXPRESSION_ERROR_SYMBOL)
        self.assertRaises(ErrorInLogicalExpression, ComputeLogicalExpression.check_logical_expression, LOGICAL_EXPRESSION_ERROR_FIRST_MEMBER_SYMBOL)
        self.assertRaises(ErrorInLogicalExpression, ComputeLogicalExpression.check_logical_expression, LOGICAL_EXPRESSION_ERROR_LAST_MEMBER_SYMBOL)
        self.assertRaises(ErrorInLogicalExpression, ComputeLogicalExpression.check_logical_expression, LOGICAL_EXPRESSION_ERROR_NODES_INTRICATE)

    def test_counting_members_logical_query(self):
        self.assertTrue(Parser.counting_members_logical_query(LOGICAL_EXPRESSION_SIMPLE_NO_ERROR) == 3)
        self.assertTrue(Parser.counting_members_logical_query(LOGICAL_EXPRESSION_INTRICATE_NO_ERROR) == 5)
        self.assertTrue(Parser.counting_members_logical_query(['B', '|', ['C', '&', 'D']]) == 5)
        self.assertTrue(Parser.counting_members_logical_query([['AKT2', '>=', '0.2'], '|', 'AKT3']) == 5)

    def test_decrease_increase_with_logical_check_raise(self):
        query = Parser.parse_query("Dec(node:AKT1) / [ A & B ] [ AKT:ON ]")
        self.assertRaises(FormulaException, FormulaChecker.check_formula, query)

        query = Parser.parse_query("Inc(state:AKT1) / [ A & B ] [ AKT:ON ]")
        self.assertRaises(FormulaException, FormulaChecker.check_formula, query)

        try:
            query = Parser.parse_query("Dec(fp:AKT1) / [ A & B ] [ AKT:ON ]")
            FormulaChecker.check_formula(query)
        except FormulaException:
            self.fail("FormulaChecker.check_formula() raised an unexpected Error")