import libsbml
import boolean
import re

algebra = boolean.BooleanAlgebra()


def booleanToSBML(node):

	if isinstance(node, boolean.boolean.Symbol):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_NAME)
		sbml_node.setName(str(node))

		return sbml_node

	elif isinstance(node, boolean.boolean.NOT):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_NOT)
		sbml_node.addChild(booleanToSBML(node.args[0]))

		return sbml_node

	elif isinstance(node, boolean.boolean.AND):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_AND)

		for arg in node.args:
			sbml_node.addChild(booleanToSBML(arg))

		return sbml_node

	elif isinstance(node, boolean.boolean.OR):
		sbml_node = libsbml.ASTNode()
		sbml_node.setType(libsbml.AST_LOGICAL_OR)

		for arg in node.args:
			sbml_node.addChild(booleanToSBML(arg))

		return sbml_node

	else:
		print("Error : Unrecognized type : %s (%s)" % (type(node), str(node)))

def addQualitativeSpecies(sbml_model, sid, species, initial_state):
	new_species = sbml_model.getPlugin("qual").createQualitativeSpecies()
	new_species.setName(species.name)
	new_species.setId(sid)
	new_species.setInitialLevel(initial_state)
	# sbml_model.getPlugin("qual").addQualitativeSpecies(new_species)

def addContinuousSpecies(sbml_model, sid, species, initial_state):
	new_species = sbml_model.createSpecies()
	new_species.setName(species.name + " (continuous)")
	new_species.setId(sid + "_cont")
	new_species.setInitialAmount(initial_state)
	new_species.setCompartment("cell")
	sbml_model.addSpecies(new_species)


def parseFormula(formula, species):
	if '?' in formula and ":" in formula:
		res_match = re.match("(.+)\?(.*):(.*)", formula)
		(formula, true, false) = tuple([element.strip() for element in res_match.groups()])
		if '@logic' in formula:
			formula = formula.replace('@logic', species.logExp)
		# print(formula)
		return (formula, true, false)
	elif isValue(species.rt_up):
		rateUp = float(species.rt_up)
		return (True, rateUp, None)
	else:
		raise Exception("formula not recognized : %s" % str(formula))

def isValue(value):
	try:
		float(value)
		return True
	except:
		return False


def getInputs(formula, species):

	if '?' in formula and ":" in formula:
		res_match = re.match("(.+)\?(.*):(.*)", formula)
		(formula, true, false) = tuple([element.strip() for element in res_match.groups()])
		if '@logic' in formula:
			formula = formula.replace('@logic', species.logExp)
		#print(formula)
		return algebra.parse(formula).symbols
	elif isValue(species.rt_up):
		rateUp = float(species.rt_up)
		return {}
	else:
		raise Exception("formula not recognized : %s" % str(formula))

def addTransition(sid, species, model):
	transition = model.getPlugin("qual").createTransition()

	output = transition.createOutput()
	output.setQualitativeSpecies(sid)
	output.setTransitionEffect(libsbml.OUTPUT_TRANSITION_EFFECT_ASSIGNMENT_LEVEL)

	inputs = getInputs(species.rt_up.replace("$", "_"), species)
	inputs.union(getInputs(species.rt_down.replace("$", "_"), species))

	for input in inputs:
		sbml_input = transition.createInput()
		sbml_input.setQualitativeSpecies(str(input))
		sbml_input.setTransitionEffect(libsbml.INPUT_TRANSITION_EFFECT_NONE)

	formula, _, _ = parseFormula(species.rt_up, species)

	terms_up = transition.createFunctionTerm()
	terms_up.setResultLevel(1)
	terms_up.setMath(booleanToSBML(algebra.parse(formula.replace('$', '_'))))

	formula, _, _ = parseFormula(species.rt_down, species)

	terms_down = transition.createFunctionTerm()
	terms_down.setResultLevel(0)
	terms_down.setMath(booleanToSBML(algebra.parse(formula.replace('$', '_'))))



def addRateValue(value):

	try:

		float_value = libsbml.ASTNode()
		float_value.setType(libsbml.AST_REAL)
		float_value.setValue(float(value))
		return float_value

	except ValueError as e:
		var_value = libsbml.ASTNode()
		var_value.setType(libsbml.AST_NAME)
		var_value.setName(value.replace("$", "_"))
		return var_value

def buildSBMLRate(rate, species):

	formula, true, false = parseFormula(rate, species)
	boolean_formula = algebra.parse(formula.replace("$", "_"))
	sbml_formula = libsbml.ASTNode()
	if isValue(formula):
		sbml_formula.setType(libsbml.AST_REAL)
		sbml_formula.setValue(float(formula))

	else:
		sbml_formula.setType(libsbml.AST_FUNCTION_PIECEWISE)
		sbml_formula.addChild(addRateValue(true))
		sbml_formula.addChild(booleanToSBML(boolean_formula))
		sbml_formula.addChild(addRateValue(false))

	return sbml_formula

def getModifiers(formula, species):
	modifiers = []

	t_formula, _, _ = parseFormula(formula, species)
	boolean_formula = algebra.parse(t_formula.replace("$", "_"))
	modifiers = [str(var) for var in boolean_formula.symbols if not str(var).startswith("_")]
	#
	# formula, _, _ = parseFormula(species.rt_down, species)
	# boolean_formula = algebra.parse(formula.replace("$", "_"))
	# modifiers += [str(var) for var in boolean_formula.symbols if not str(var).startswith("_")]

	return list(set(modifiers))

def addReaction(sbml_model, sid, species):

	new_reaction = sbml_model.createReaction()
	# Here reactions are simple : reversible synthesis/degradation
	new_reaction.setId("production_%s" % sid)
	new_reaction.setReversible(False)
	product = new_reaction.createProduct()
	product.setSpecies(sid)

	modifiers = getModifiers(species.rt_up, species)
	for modifier in modifiers:
		new_modifier = new_reaction.createModifier()
		new_modifier.setSpecies(modifier)

	up_formula = buildSBMLRate(species.rt_up, species)

	kinetic_law = new_reaction.createKineticLaw()
	kinetic_law.setMath(up_formula)
	# new_reaction.setKineticLaw(sbml_formula)
	sbml_model.addReaction(new_reaction)

	new_reaction = sbml_model.createReaction()
	new_reaction.setId("degradation_%s" % sid)
	new_reaction.setReversible(False)
	reactant = new_reaction.createReactant()
	reactant.setSpecies(sid)

	modifiers = getModifiers(species.rt_down, species)
	for modifier in modifiers:
		new_modifier = new_reaction.createModifier()
		new_modifier.setSpecies(modifier)

	down_formula = buildSBMLRate(species.rt_down, species)
	kinetic_law = new_reaction.createKineticLaw()
	kinetic_law.setMath(down_formula)
	sbml_model.addReaction(new_reaction)


def addParameter(sbml_model, name, value):

	new_param = sbml_model.createParameter()
	new_param.setName(name[1: len(name)])
	new_param.setId("_%s" % name[1: len(name)])
	new_param.setValue(value)
	sbml_model.addParameter(new_param)

def to_sbml(model, filename):

	sbml_doc = libsbml.SBMLDocument(3, 1)
	sbml_doc.enablePackage("http://www.sbml.org/sbml/level3/version1/qual/version1", "qual", True)
	sbml_doc.setPackageRequired("qual", True)
	sbml_model = sbml_doc.createModel()

	compartment = sbml_model.createCompartment()
	compartment.setName("Cell")
	compartment.setId("cell")

	initial_state = model.get_initial_state()

	for sid, species in model.network.items():
		addQualitativeSpecies(sbml_model, sid, species, initial_state[sid])
		addTransition(sid, species, sbml_model)

		# addContinuousSpecies(sbml_model, sid, species, initial_state[sid])
		# addReaction(sbml_model, sid, species)

	for param, value in model.param.items():
		if param.startswith("$"):
			addParameter(sbml_model, param, float(value))

	libsbml.writeSBMLToFile(sbml_doc, filename)
	return sbml_doc