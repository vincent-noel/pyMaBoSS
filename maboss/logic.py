"""utilitary functions to manipulate boolean expressions."""

from __future__ import print_function
import pyparsing as pp


class ASTNode(object):
    def __init__(self, tokens):
        self.tokens = [token for token in tokens if token not in ['(', ')']]
        self.__dict__ = self.assignFields()

    def __str__(self):
        return str(self.__dict__)

    __repr__ = __str__


class ASTVar(ASTNode):
    def assignFields(self):
        self.operator = 'var'
        self.args = self.tokens[0]
        del self.tokens
        return self.__dict__


class ASTNot(ASTNode):
    def assignFields(self):
        if len(self.tokens) < 2:
            return self.tokens[0].__dict__

        self.operator = 'not'
        self.args = [self.tokens[1]]
        del self.tokens
        return self.__dict__


class ASTAnd(ASTNode):
    def assignFields(self):
        if len(self.tokens) < 2:
            return self.tokens[0].__dict__

        self.operator = 'and'
        self.args = self.tokens[::2]
        del self.tokens
        return self.__dict__


class ASTOr(ASTNode):
    def assignFields(self):
        if len(self.tokens) < 2:
            return self.tokens[0].__dict__

        self.args = self.tokens[::2]
        self.operator = 'or'
        del self.tokens
        return self.__dict__


class ASTXor(ASTNode):
    def assignFields(self):
        if len(self.tokens) < 2:
            return self.tokens[0].__dict__

        self.args = self.tokens[::2]
        self.operator = 'xor'
        del self.tokens
        return self.__dict__


class ASTIFE(ASTNode):
    def assignFields(self):
        if len(self.tokens) < 5:
            return self.tokens[0].__dict__

        self.operator = 'ife'
        self.args = self.tokens[::2]
        # self.values = self.tokens[2::2]
        del self.tokens
        return self.__dict__


logExp = pp.Forward()
boolCst = pp.oneOf("True False")
boolNot = pp.oneOf("! ~ NOT")
boolAnd = pp.oneOf("&& & AND")
boolOr = pp.oneOf("|| | OR")
boolXor = pp.oneOf("^ XOR")
boolTest = pp.Literal("?")
boolElse = pp.Literal(":")
varName = (~boolAnd + ~boolOr + ~boolXor + ~boolNot + ~boolCst + ~boolTest + ~boolElse
           + ~pp.Literal('Node') + pp.Word(pp.alphas+'$', pp.alphanums+'_'))
varName.setParseAction(ASTVar)
lparen = '('
rparen = ')'
# logTerm = (pp.Optional(boolNot)
#            + (boolCst | varName | (lparen + logExp + rparen)))
logTerm = (boolCst | varName | (lparen + logExp + rparen))

logNot = (pp.ZeroOrMore(boolNot) + logTerm).setResultsName("AST").setParseAction(ASTNot)
logAnd = (logNot + pp.ZeroOrMore(boolAnd + logNot)).setResultsName("AST").setParseAction(ASTAnd)
logOr = (logAnd + pp.ZeroOrMore(boolOr + logAnd)).setResultsName("AST").setParseAction(ASTOr)
logXor = (logOr + pp.ZeroOrMore(boolXor + logOr)).setResultsName("AST").setParseAction(ASTXor)
logIFE = (logXor + pp.ZeroOrMore(boolTest + logXor + boolElse + logXor)).setResultsName("AST").setParseAction(ASTIFE)
logExp << pp.Combine(logIFE, adjacent=False, joinString=' ')



def _check_logic_syntax(string):
    """Return True iff string is a syntaxically correct boolean expression."""

    print("")
    print(logExp.parseString(string).asList())
    print(logExp.parseString(string).asDict())
    return logExp.matches(string)


def _check_logic_defined(name_list, logic_list):
    """Check if the list of logic is consistant.

    Return True iff all expression in logic_list are syntaxically correct and
    all contains only variables present in name_list.
    """
    _check_logic_defined.failed = False

    # We modify the behaviour of boolExp.parseString so that the parsing also
    # check if variables exist.
    def check_var(var):
        if var[0] not in name_list:
            print("Error: unkown variable %s" % var[0], file=stderr)
            _check_logic_defined.failed = True
        return var

    varName.setParseAction(check_var)

    for string in logic_list:
        if not _check_logic_syntax(string):
            print("Error: syntax error %s" % string, file=stderr)
            return False
        if _check_logic_defined.failed:
            varName.setParseAction(lambda x: x)
            return False

    varName.setParseAction(lambda x: x)
    return True
