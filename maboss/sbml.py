import libsbml
import re
from .ast import logExp as logExpAST
from .ast import ASTReal, ASTVar, ASTBool, ASTNot, ASTAnd, ASTOr, ASTXor, ASTIFE

uri_qual = 'http://www.sbml.org/sbml/level3/version1/qual/version1'
uri_maboss = 'http://www.sbml.org/sbml/level3/version1/maboss/version1'


def booleanToSBML(node, forceBoolean=False):

	if isinstance(node, ASTReal):
		sbml_node = libsbml.ASTNode()
		if forceBoolean:
			if node.args == 1.0:
				sbml_node.setType(libsbml.AST_CONSTANT_TRUE)
			else:
				sbml_node.setType(libsbml.AST_CONSTANT_FALSE)
		else:
			sbml_node.setType(libsbml.AST_REAL)
			sbml_node.setValue(float(node.args[0]))

		return sbml_node

	elif isinstance(node, ASTBool):
		sbml_node = libsbml.ASTNode()

		if node.args[0] == 'TRUE':
			sbml_node.setType(libsbml.AST_CONSTANT_TRUE)
		elif node.args[0] == 'FALSE':
			sbml_node.setType(libsbml.AST_CONSTANT_FALSE)
		else:
			print("Unrecognized boolean value : %s" % node.args[0])

		return sbml_node

	elif isinstance(node, ASTVar):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_NAME)
		sbml_node.setName(node.args[0].replace('$', '_'))

		if forceBoolean:
			sbml_one = libsbml.ASTNode()
			sbml_one.setType(libsbml.AST_REAL)
			sbml_one.setValue(1)

			sbml_eq = libsbml.ASTNode()
			sbml_eq.setType(libsbml.AST_RELATIONAL_EQ)
			sbml_eq.addChild(sbml_node)
			sbml_eq.addChild(sbml_one)

			return sbml_eq
		else:
			return sbml_node

	elif isinstance(node, ASTNot):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_NOT)
		sbml_node.addChild(booleanToSBML(node.args[0], forceBoolean))
		return sbml_node

	elif isinstance(node, ASTAnd):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_AND)

		for arg in node.args:
			sbml_node.addChild(booleanToSBML(arg, forceBoolean))

		return sbml_node

	elif isinstance(node, ASTOr):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_OR)

		for arg in node.args:
			sbml_node.addChild(booleanToSBML(arg, forceBoolean))

		return sbml_node

	elif isinstance(node, ASTXor):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_XOR)

		for arg in node.args:
			sbml_node.addChild(booleanToSBML(arg, forceBoolean))

		return sbml_node

	elif isinstance(node, ASTIFE):

		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_FUNCTION_PIECEWISE)

		sbml_node.addChild(booleanToSBML(node.args[1], forceBoolean))
		sbml_node.addChild(booleanToSBML(node.args[0], forceBoolean))
		sbml_node.addChild(booleanToSBML(node.args[2], forceBoolean))

		return sbml_node

	else:
		print("Error : Unrecognized type : %s (%s)" % (type(node), str(node)))

def addQualitativeSpecies(sbml_model, sid, species, initial_state):
	new_species = sbml_model.getPlugin("qual").createQualitativeSpecies()
	new_species.setName(species.name)
	new_species.setId(sid)
	new_species.setInitialLevel(initial_state)
	new_species.setCompartment("cell")
	new_species.setConstant(False)
	# sbml_model.getPlugin("qual").addQualitativeSpecies(new_species)

def getAST(formula):
	return logExpAST.parseString(formula).asDict()['AST']

def getInputs(formula, species):

	i_var = {("@%s" % key): getAST(value) for key, value in species.internal_var.items()}
	developped_res = getAST(formula).develop(getAST(species.logExp), i_var)
	list_vars = [var for var in developped_res.listVars() if not var.startswith('$')]
	return set(list_vars)

def addTransition(model, sid, species):
	transition = model.getPlugin("qual").createTransition()
	transition.setId("transitions_%s" % species.name)

	output = transition.createOutput()
	output.setQualitativeSpecies(sid)
	output.setTransitionEffect(libsbml.OUTPUT_TRANSITION_EFFECT_ASSIGNMENT_LEVEL)

	inputs = getInputs(species.rt_up, species)
	inputs.union(getInputs(species.rt_down, species))

	for input in inputs:
		sbml_input = transition.createInput()
		sbml_input.setQualitativeSpecies(str(input))
		sbml_input.setTransitionEffect(libsbml.INPUT_TRANSITION_EFFECT_NONE)

	internal_var = {("@%s" % key): getAST(value) for key, value in species.internal_var.items()}

	terms_up = transition.createFunctionTerm()
	terms_up.setResultLevel(1)

	dev_formula_up = getAST(species.rt_up).develop(getAST(species.logExp), internal_var)
	sbml_formula = booleanToSBML(dev_formula_up)
	terms_up.setMath(sbml_formula)

	terms_down = transition.createFunctionTerm()
	terms_down.setResultLevel(0)

	dev_formula_down = getAST(species.rt_down).develop(getAST(species.logExp), internal_var)
	sbml_formula = booleanToSBML(dev_formula_down)
	terms_down.setMath(sbml_formula)

	default = transition.createDefaultTerm()
	default.setResultLevel(0)

def findTransition(xml_model, id):

	transitions = xml_model.getChild('listOfTransitions')
	i = 0
	while i < transitions.getNumChildren():
		t_child = transitions.getChild(i)

		if t_child.hasAttr('id', uri_qual) and t_child.getAttrValue('id', uri_qual) == id:
			break
		i += 1

	if i < transitions.getNumChildren():
		return transitions.getChild(i)
	else:
		print("Unable to find transition with id == %s" % ('transitions_%s' % id))

def findFunction(xml_transition, resultLevel):

	functions = xml_transition.getChild('listOfFunctionTerms')
	j = 0
	while j < functions.getNumChildren():
		t_function = functions.getChild(j)
		if t_function.hasAttr('resultLevel', uri_qual) and t_function.getAttrValue('resultLevel', uri_qual) == resultLevel:
			break
		j += 1

	if j < functions.getNumChildren():

		return functions.getChild(j)
	else:
		print("unable to find function with resultLevel == %s" % resultLevel)

def addMaBoSSMathToTransitions(xml_model, sid, species):

	transition = findTransition(xml_model, "transitions_%s" % species.name)

	function_activation = findFunction(transition, "1")
	maboss_mathml = function_activation.getChild(0).clone()
	maboss_mathml.setTriple(libsbml.XMLTriple("math", uri_maboss, "maboss"))
	function_activation.addChild(maboss_mathml)

	function_inactivation = findFunction(transition, "0")
	maboss_mathml = function_inactivation.getChild(0).clone()
	maboss_mathml.setTriple(libsbml.XMLTriple("math", uri_maboss, "maboss"))
	function_inactivation.addChild(maboss_mathml)

def findSBMLTransitionFunction(transition, resultLevel):

	i = 0
	while i < transition.getNumFunctionTerms():
		if transition.getFunctionTerm(i).getResultLevel() == resultLevel:
			break
		i += 1

	if i < transition.getNumFunctionTerms():
		return transition.getFunctionTerm(i)
	else:
		print("Unable to find the function term with resultLevel == %s" % resultLevel)

def forceBooleanMath(formula):

	if formula.operator == 'ife':
		return formula.args[0]

	elif formula.operator == 'bool':
		return formula.args[0]

	elif formula.operator == 'real':
		if formula.args[0] == 0:
			return ASTBool(['false'])
		else:
			return ASTBool(['true'])

	return formula

def switchToBooleanTransitions(sbml_model, sid, species):

	internal_var = {("@%s" % key): getAST(value) for key, value in species.internal_var.items()}
	transition = sbml_model.getPlugin("qual").getElementBySId("transitions_%s" % species.name)

	activation = findSBMLTransitionFunction(transition, 1)
	dev_formula_up = getAST(species.rt_up).develop(getAST(species.logExp), internal_var)
	activation.setMath(booleanToSBML(forceBooleanMath(dev_formula_up), forceBoolean=True))

	inactivation = findSBMLTransitionFunction(transition, 0)
	dev_formula_down = getAST(species.rt_down).develop(getAST(species.logExp), internal_var)
	inactivation.setMath(booleanToSBML(forceBooleanMath(dev_formula_down), forceBoolean=True))

def addParameter(sbml_model, name, value):

	new_param = sbml_model.createParameter()
	new_param.setName(name[1: len(name)])
	new_param.setId("_%s" % name[1: len(name)])
	new_param.setValue(value)
	new_param.setConstant(True)
	sbml_model.addParameter(new_param)

def to_sbml(model, filename, maboss_extras=True):

	sbml_doc = libsbml.SBMLDocument(3, 1)
	sbml_doc.enablePackage("http://www.sbml.org/sbml/level3/version1/qual/version1", "qual", True)
	sbml_doc.setPackageRequired("qual", True)
	sbml_model = sbml_doc.createModel()

	compartment = sbml_model.createCompartment()
	compartment.setName("Cell")
	compartment.setId("cell")
	compartment.setConstant(True)

	initial_state = model.get_initial_state()

	for sid, species in model.network.items():
		addQualitativeSpecies(sbml_model, sid, species, initial_state[sid])
		addTransition(sbml_model, sid, species)

	for param, value in model.param.items():
		if param.startswith("$"):
			addParameter(sbml_model, param, float(value))

	if maboss_extras:
		xml_doc = sbml_doc.toXMLNode()
		xml_doc.addAttr(libsbml.XMLTriple("maboss", uri_maboss, "xmlns"), uri_maboss)
		xml_doc.addAttr(libsbml.XMLTriple("required", uri_maboss, "maboss"), "true")

		xml_model = xml_doc.getChild('model')

		for sid, species in model.network.items():
			addMaBoSSMathToTransitions(xml_model, sid, species)


		sbml_doc = libsbml.readSBMLFromString(xml_doc.toXMLString())
		sbml_model = sbml_doc.getModel()

	for sid, species in model.network.items():
		switchToBooleanTransitions(sbml_model, sid, species)

	libsbml.writeSBMLToFile(sbml_doc, filename)
	return sbml_doc